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
def my_big_numpy_array():
    prob_pos = np.linspace(0, 1, 1000)
    prob_neg = 1 - prob_pos
    arr = np.vstack([prob_neg, prob_pos]).T
    original_arr = arr.copy()

    # Now let's make it perfectly imperfect - it needs to get 90% of [0-0.1) wrong, 80% of [0.1,0.2), etc.
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(
            original_arr > bin_lower.item(), original_arr <= bin_upper.item()
        )
        to_fix_arr = arr[in_bin[:, 1]]
        n_to_fix = int(to_fix_arr.shape[0] * bin_upper)
        to_fix_arr[:n_to_fix] = [[0, 1]]
        to_fix_arr[n_to_fix:] = [[1, 0]]
        arr[in_bin[:, 1]] = to_fix_arr
    assert np.abs(arr[:, 1].sum() - 549) < 0.01
    return arr


@pytest.fixture
def my_labels(my_numpy_array):
    labels = (my_numpy_array[:, 1] >= 0.5) * 1
    return labels


@pytest.fixture
def my_perfect_labels(my_big_numpy_array):
    perfect_labels = np.zeros((my_big_numpy_array.shape[0], 1))

    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(
            my_big_numpy_array > bin_lower.item(),
            my_big_numpy_array <= bin_upper.item(),
        )
        to_fix_labels = perfect_labels[in_bin[:, 1]]
        n_to_fix = int(to_fix_labels.shape[0] * bin_upper)
        to_fix_labels[:n_to_fix] = 1
        perfect_labels[in_bin[:, 1]] = to_fix_labels

    return perfect_labels


def test_expected_calibration_error(my_numpy_array, my_labels):
    ece = mc.expected_calibration_error(my_numpy_array, my_labels, 10)
    assert np.abs(ece - 0.25) < 0.03


def test_expected_calibration_error_perfect(my_big_numpy_array, my_perfect_labels):
    ece = mc.expected_calibration_error(my_big_numpy_array, my_perfect_labels, 10)
    assert np.abs(ece) < 0.03
