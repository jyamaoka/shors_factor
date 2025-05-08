import pytest
from shors_factor.factoring_simple import factor_number

def test_factor_known_numbers():
    # Note: Shor's algorithm is probabilistic
    known_factors = {
        15: [3, 5],
        21: [3, 7],
        35: [5, 7],
    }

    for number, expected_factors in known_factors.items():
        factor = factor_number(number, max_attempts=200)
        assert factor == expected_factors, f"{number} -> {factor}"

def test_invalid_number():
    with pytest.raises(ValueError):
        factor_number(1)

    with pytest.raises(ValueError):
        factor_number(256)
