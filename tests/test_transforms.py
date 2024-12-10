# test_transforms.py

import numpy as np
import pytest

from src.transforms import (
    affine_transform,
    augment_to_3d,
    composite_transform,
    min_max_scale,
    normalize,
    polynomial_features,
    rotate2D,
    scale,
    shift,
)


# Test affine_transform
@pytest.mark.parametrize(
    "features, A, b, expected",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.eye(2),
            np.zeros(2),
            np.array([[1, 2], [3, 4]]),
        ),  # Identity
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[2, 0], [0, 2]]),
            np.zeros(2),
            np.array([[2, 4], [6, 8]]),
        ),  # Scaling
        (
            np.array([[1, 2], [3, 4]]),
            np.eye(2),
            np.array([1, -1]),
            np.array([[2, 1], [4, 3]]),
        ),  # Translation
    ],
    ids=["identity", "scaling", "translation"],
)
def test_affine_transform(features, A, b, expected):
    result = affine_transform(features, A, b)
    assert np.allclose(result, expected)


# Test normalize
def test_normalize():
    features = np.array([[1, 2], [3, 4]])
    expected = np.array([[-1, -1], [1, 1]])  # Normalized around 0 mean, unit variance
    result = normalize(features)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


def test_normalize_zero_variance_avoid_div_by_zero():
    features = np.array([[1, 1], [1, 1]])
    expected = np.array([[0, 0], [0, 0]])  # Zero variance
    result = normalize(features)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


# Test min_max_scale
def test_min_max_scale():
    features = np.array([[1, 2], [3, 4]])
    expected = np.array([[0, 0], [1, 1]])  # Scaled to [0, 1]
    result = min_max_scale(features)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


def test_min_max_scale_zero_range_avoid_div_by_zero():
    features = np.array([[1, 1], [1, 1]])
    expected = np.array([[0, 0], [0, 0]])  # Zero range
    result = min_max_scale(features)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


# Test polynomial_features
@pytest.mark.parametrize(
    "features, degree, expected",
    [
        (np.array([[1, 2]]), 2, np.array([[1, 1, 2, 1, 2, 4]])),  # Degree 2
        (np.array([[1, 2]]), 3, np.array([[1, 1, 2, 1, 2, 4, 1, 2, 4, 8]])),  # Degree 3
    ],
    ids=["degree_2", "degree_3"],
)
def test_polynomial_features(features, degree, expected):
    result = polynomial_features(features, degree)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


# Test rotate2D
@pytest.mark.parametrize(
    "features, angle, expected",
    [
        (np.array([[1, 0]]), np.pi / 2, np.array([[0, 1]])),  # 90 degrees
        (np.array([[0, 1]]), np.pi, np.array([[0, -1]])),  # 180 degrees
        (np.array([[1, 0]]), -np.pi / 2, np.array([[0, -1]])),  # -90 degrees
    ],
    ids=["90_degrees", "180_degrees", "-90_degrees"],
)
def test_rotate2D(features, angle, expected):
    result = rotate2D(features, angle)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


# 6. Test augment_to_3d
def test_augment_to_3d():
    features = np.array([[1, 2]])
    constant = 1
    expected = np.array([[1, 1, 2]])  # Add constant dimension
    result = augment_to_3d(features, constant)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


# Test scale
def test_scale():
    features = np.array([[1, 2], [3, 4]])
    scale_factor = 2
    expected = np.array([[2, 4], [6, 8]])  # Scaled by 2
    result = scale(features, scale_factor)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


# Test shift
@pytest.mark.parametrize(
    "features, offset, expected",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([1, -1]),
            np.array([[2, 1], [4, 3]]),
        ),  # Shift
        (
            np.array([[1, 2], [3, 4]]),
            np.array([-1, 2]),
            np.array([[0, 4], [2, 6]]),
        ),  # Negative offset
    ],
    ids=["shift", "negative_offset"],
)
def test_shift(features, offset, expected):
    result = shift(features, offset)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


# Test composite_transform
def test_composite_transform():
    features = np.array([[1, 2]])
    transforms = [
        lambda x: x + 1,  # Add 1 to each element
        lambda x: x * 2,  # Multiply each element by 2
    ]
    expected = np.array([[4, 6]])
    result = composite_transform(features, transforms)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}."
