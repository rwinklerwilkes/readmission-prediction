import numpy as np
import pytest
from production import model_calibration as mc


@pytest.fixture
def my_numpy_array():
    prob_pos = np.linspace(0, 1, 100)
    prob_neg = 1 - prob_pos
    arr = np.vstack([prob_neg, prob_pos]).T
    return arr


@pytest.fixture
def my_labels(my_numpy_array):
    labels = (my_numpy_array[:, 1] >= 0.5) * 1
    return labels


def test_expected_calibration_error(my_numpy_array, my_labels):
    ece = mc.expected_calibration_error(my_numpy_array, my_labels, 10)
    assert np.abs(ece - 0.25) < 0.03
