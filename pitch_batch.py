"""Parsing and validation for ordered pitch-variant generation."""

MAX_PITCH_VARIANTS = 25


class PitchBatchError(ValueError):
    pass


def parse_pitch_spec(spec, minimum=-36, maximum=36, limit=MAX_PITCH_VARIANTS):
    """Parse comma-separated integers and inclusive `start:end[:step]` ranges."""
    if spec is None or not str(spec).strip():
        raise PitchBatchError("Enter at least one pitch for batch generation")

    pitches = []
    seen = set()
    for token in (part.strip() for part in str(spec).split(",")):
        if not token:
            continue
        if ":" in token:
            values = _parse_range(token, limit)
        else:
            values = (_parse_integer(token),)
        for pitch in values:
            if pitch < minimum or pitch > maximum:
                raise PitchBatchError(f"Pitch values must be between {minimum} and {maximum} semitones")
            if pitch in seen:
                continue
            seen.add(pitch)
            pitches.append(pitch)
            if len(pitches) > limit:
                raise PitchBatchError(f"A batch may contain at most {limit} pitch variants")

    if not pitches:
        raise PitchBatchError("Enter at least one pitch for batch generation")
    return pitches


def _parse_integer(value):
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise PitchBatchError(f"Invalid pitch value: {value}") from error


def _parse_range(token, limit):
    parts = token.split(":")
    if len(parts) not in (2, 3):
        raise PitchBatchError(f"Invalid pitch range: {token}")
    start, end = (_parse_integer(parts[0]), _parse_integer(parts[1]))
    step = _parse_integer(parts[2]) if len(parts) == 3 else (1 if end >= start else -1)
    if step == 0:
        raise PitchBatchError("Pitch range step cannot be zero")
    if (end - start) * step < 0:
        raise PitchBatchError(f"Pitch range step moves away from its end: {token}")
    count = abs(end - start) // abs(step) + 1
    if count > limit:
        raise PitchBatchError(f"A batch may contain at most {limit} pitch variants")
    stop = end + (1 if step > 0 else -1)
    return range(start, stop, step)
