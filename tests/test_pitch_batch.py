import pytest

from pitch_batch import parse_pitch_spec


def test_pitch_batch_accepts_values_and_inclusive_ranges():
    assert parse_pitch_spec("-12,-7,0,7,12", -36, 36) == [-12, -7, 0, 7, 12]
    assert parse_pitch_spec("-4:4:2", -36, 36) == [-4, -2, 0, 2, 4]
    assert parse_pitch_spec("4:-4:-2", -36, 36) == [4, 2, 0, -2, -4]


@pytest.mark.parametrize("value", ["", "1:2:0", "40", "a", "1:4:-1"])
def test_pitch_batch_rejects_invalid_specs(value):
    with pytest.raises(ValueError):
        parse_pitch_spec(value, -36, 36)


def test_pitch_batch_rejects_huge_range_without_materializing_it():
    with pytest.raises(ValueError, match="at most 25"):
        parse_pitch_spec("-1000000000:1000000000", -36, 36)
