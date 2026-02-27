import argparse
import base64
import json
import os.path
import re
import subprocess
import traceback

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema.exceptions import ValidationError

ARCHITECTURE_NAME = 'so_vits_svc_3'
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, 'so_vits_svc_3')

RAW_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'raw')
OUTPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'results')
INFERENCE_TEMPLATE_PATH = os.path.join(ARCHITECTURE_ROOT, 'inference_main_template.py')
INFERENCE_CODE_PATH = os.path.join(ARCHITECTURE_ROOT, 'inference_main.py')
TEMP_FILE_EXTENSION = '.flac'

PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, '.venvs', 'so_vits_svc_3', 'bin', 'python')

app = Flask(__name__)


def register_methods(cache):
    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            input_filename_sans_extension, character, pitch_shift, output_filename_sans_extension, gpu_id, \
                session_id = parse_inputs()
            ensure_template_exists()
            modify_inference_file(input_filename_sans_extension, character, pitch_shift, session_id)
            copy_input_audio(input_filename_sans_extension, session_id)
            execute_program(gpu_id)
            copy_output(output_filename_sans_extension, session_id)
            hsc.clean_up(get_temp_files())
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = hsc.construct_full_error_message(ARCHITECTURE_ROOT, get_temp_files())

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    @app.route('/gpu-info', methods=['GET'])
    def get_gpu_info():
        return hsc.get_gpu_info_from_another_venv(PYTHON_EXECUTABLE)

    def parse_inputs():
        schema = {
            'type': 'object',
            'properties': {
                'Inputs': {
                    'type': 'object',
                    'properties': {
                        'User Audio': {'type': 'string'}
                    },
                    'required': ['User Audio']
                },
                'Options': {
                    'type': 'object',
                    'properties': {
                        'Architecture': {'type': 'string'},
                        'Character': {'type': 'string'},
                        'Pitch Shift': {'type': 'integer'},
                    },
                    'required': ['Character', 'Pitch Shift']
                },
                'Output File': {'type': 'string'},
                'GPU ID': {'type': ['string', 'integer']},
                'Session ID': {'type': ['string', 'null']}
            },
            'required': ['Inputs', 'Options', 'Output File', 'GPU ID', 'Session ID']
        }

        try:
            jsonschema.validate(instance=request.json, schema=schema)
        except ValidationError as e:
            raise BadInputException(e.message)

        input_filename_sans_extension = request.json['Inputs']['User Audio']
        character = request.json['Options']['Character']
        pitch_shift = request.json['Options']['Pitch Shift']
        output_filename_sans_extension = request.json['Output File']
        gpu_id = request.json['GPU ID']
        session_id = request.json['Session ID']
        return input_filename_sans_extension, character, pitch_shift, output_filename_sans_extension, gpu_id, session_id

    class BadInputException(Exception):
        pass

    def ensure_template_exists():
        """The very first time that generate() is called, we make a copy of inference_main.py. From there on out, we
        read from the copy and write modified contents to inference_main.py. That way, if inference_main.py ever gets
        corrupted (e.g. The program terminates unexpectedly while writing to the file), it won't put the application in
        a bad state. We will automatically reconstruct inference_main.py from the copy the next time generate() is
        called."""
        if not os.path.isfile(INFERENCE_TEMPLATE_PATH):
            with open(INFERENCE_CODE_PATH, 'r') as file:
                content = file.read()
            with open(INFERENCE_TEMPLATE_PATH, 'w') as file:
                file.write(content)

    def modify_inference_file(input_filename_sans_extension, character, pitch_shift, session_id):
        with open(INFERENCE_TEMPLATE_PATH, 'r') as file:
            content = file.read()
        modified_content = modify_content(content, input_filename_sans_extension, character, pitch_shift, session_id)
        with open(INFERENCE_CODE_PATH, 'w') as file:
            file.write(modified_content)

    def modify_content(content, input_filename_sans_extension, character, pitch_shift, session_id):
        model_path_line, config_path_line, clean_names_line, trans_line, speaker_line = \
            construct_lines(input_filename_sans_extension, character, pitch_shift, session_id)
        content = re.sub(r'^model_path = .*', model_path_line, content, flags=re.M)
        content = re.sub(r'^config_path = .*', config_path_line, content, flags=re.M)
        content = re.sub(r'^clean_names = .*', clean_names_line, content, flags=re.M)
        content = re.sub(r'^trans = .*', trans_line, content, flags=re.M)
        content = re.sub(r'^spk_list = .*', speaker_line, content, flags=re.M)
        return content

    def construct_lines(input_filename_sans_extension, character, pitch_shift, session_id):
        model_path_line, config_path_line = construct_model_and_config_path_lines(character)
        clean_names_line = construct_clean_names_line(input_filename_sans_extension, session_id)
        trans_line = construct_trans_line(pitch_shift)
        speaker_line = construct_speaker_line(character)
        return model_path_line, config_path_line, clean_names_line, trans_line, speaker_line

    def construct_model_and_config_path_lines(character):
        model_path, config_path = get_model_and_config_paths(character)
        model_path_line = 'model_path = "' + model_path + '"'
        config_path_line = 'config_path = "' + config_path + '"'
        return model_path_line, config_path_line

    def get_model_and_config_paths(character):
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        model_filename, config_filename = get_model_and_config_filenames(character_dir)
        model_path = os.path.join(character_dir, model_filename)
        config_path = os.path.join(character_dir, config_filename)
        return model_path, config_path

    def get_model_and_config_filenames(character_dir):
        return get_model_filename(character_dir), get_config_filename(character_dir)

    def get_config_filename(character_dir):
        potential_name = os.path.join(character_dir, 'config.json')
        if not os.path.isfile(potential_name):
            raise Exception('Config file not found! Expecting a file with the name config.json in ' + character_dir)
        else:
            return potential_name

    def get_model_filename(character_dir):
        potential_names = [file for file in os.listdir(character_dir) if file.startswith('G_')]
        if len(potential_names) == 0:
            raise Exception('Model file was not found! Expected a file with the name G_<number>.pth in ' +
                            character_dir)
        if len(potential_names) > 1:
            raise Exception('Too many model files found! Expected only one file with the name G_<number>.pth in '
                            + character_dir)
        else:
            return potential_names[0]

    def construct_clean_names_line(input_filename_sans_extension, session_id):
        check_file_exists(input_filename_sans_extension, session_id)
        return 'clean_names = ["' + input_filename_sans_extension + TEMP_FILE_EXTENSION + '"]'

    def check_file_exists(input_filename_sans_extension, session_id):
        file_exists = cache.file_is_already_cached(Stage.PREPROCESSED, session_id, input_filename_sans_extension)
        if not file_exists:
            raise Exception('Input audio not found! Expected a file named "' + input_filename_sans_extension +
                            '" in the Preprocess cache.')

    def construct_trans_line(pitch_shift):
        try:
            int(str(pitch_shift))
        except ValueError:
            raise Exception('The specified pitch shift, ' + str(pitch_shift) + ' should be an integer value, '
                            'e.g. -5 or 11')
        else:
            return 'trans = [' + str(pitch_shift) + ']'

    def construct_speaker_line(character):
        speaker = get_speaker(character)
        return 'spk_list = ["' + speaker + '"]'

    def get_speaker(character):
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        config_filename = get_config_filename(character_dir)
        with open(config_filename, 'r') as file:
            config_json = json.load(file)
        speaker_dict = config_json['spk']
        speaker = get_speaker_key(character_dir, speaker_dict)
        return speaker

    def get_speaker_key(character_dir, speaker_dict):
        all_speakers = speaker_dict.keys()
        if len(all_speakers) == 1:
            return list(all_speakers)[0]
        else:
            selected_speaker = get_speaker_from_speaker_config(character_dir)
            if selected_speaker not in all_speakers:
                raise Exception("The key \"" + selected_speaker + "\", from speaker.json, not found in config.json. "
                                                                  "Expecting one of: " + str(list(all_speakers)))
            else:
                return selected_speaker

    def get_speaker_from_speaker_config(character_dir):
        potential_json_path = os.path.join(character_dir, 'speaker.json')
        if not os.path.isfile(potential_json_path):
            raise Exception("speaker.json not found! If config.json has more than one speaker, then you must add a "
                            "speaker.json file to the character folder which specifies the desired speaker. The "
                            "contents of speaker.json should be a single entry in the following format: "
                            "{\"speaker\": <desired speaker name>}")
        else:
            with open(potential_json_path, 'r') as file:
                speaker_selector = json.load(file)
            return speaker_selector['speaker']

    def copy_input_audio(input_filename_sans_extension, session_id):
        data, sr = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_filename_sans_extension)
        target = os.path.join(RAW_COPY_FOLDER, input_filename_sans_extension + TEMP_FILE_EXTENSION)
        try:
            soundfile.write(target, data, sr)
        except Exception as e:
            raise Exception("Unable to copy file from Hay Say's audio cache to rvc's raw directory.") from e

    def execute_program(gpu_id):
        env = hsc.select_hardware(gpu_id)
        subprocess.run([PYTHON_EXECUTABLE, INFERENCE_CODE_PATH], env=env, cwd=ARCHITECTURE_ROOT)

    def copy_output(output_filename_sans_extension, session_id):
        filename = get_output_filename()
        source_path = os.path.join(OUTPUT_COPY_FOLDER, filename)
        array_output, sr_output = hsc.read_audio(source_path)
        cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array_output, sr_output)

    def get_output_filename():
        all_filenames = [file for file in os.listdir(OUTPUT_COPY_FOLDER)]
        if len(all_filenames) == 0:
            raise Exception('No output file was produced! Expected file to appear in ' + OUTPUT_COPY_FOLDER)
        elif len(all_filenames) > 1:
            message = 'More than one file was found in ' + OUTPUT_COPY_FOLDER + '! Please alert the maintainers of ' \
                      'Hay Say; they should be cleaning that directory every time output is generated. '
            try:
                hsc.clean_up(get_temp_files())
            except Exception as e:
                raise Exception(message + 'An attempt was made to clean the directory to correct this situation, but '
                                          'the operation failed.') from e
            raise Exception(message + 'The directory has now been cleaned. Please try generating your output again.')
        else:
            return all_filenames[0]

    def get_temp_files():
        output_files_to_clean = [os.path.join(OUTPUT_COPY_FOLDER, file) for file in os.listdir(OUTPUT_COPY_FOLDER)]
        input_files_to_clean = [os.path.join(RAW_COPY_FOLDER, file) for file in os.listdir(RAW_COPY_FOLDER)]
        return output_files_to_clean + input_files_to_clean


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='A webservice interface for voice conversion with so-vits-svc 3.0')
    parser.add_argument('--cache_implementation', default='file', choices=hsc.cache_implementation_map.keys(),
                        help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = hsc.select_cache_implementation(args.cache_implementation)
    register_methods(cache)
    app.run(host='0.0.0.0', port=6575)
