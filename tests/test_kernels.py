import numpy as np
import pytest

from src.kernels import (
    affine_kernel,
    exponential_kernel,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    quadratic_kernel,
    rbf_gaussian_kernel,
)


# TODO: Should parameterize these test for x and y datatypes (ints, floats, numpy arrays, etc.)?
# TODO: Should we parameterize these test classes for each kernel function?
class TestLinearKernel:
    def test_basic_ints(self):
        x = [1, 2]
        y = [3, 4]
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"

    def test_basic_floats(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"

    def test_basic_mixed_floats_and_ints(self):
        x = [1, 2]
        y = [3.0, 4.0]
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"

    def test_basic_numpy_ints(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"

    def test_basic_numpy_floats(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"

    def test_basic_numpy_mixed_floats_and_ints(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"


class TestAffineKernel:
    def test_basic_ints_with_default_c(self):
        x = [1, 2]
        y = [3, 4]
        expected = 1 * 3 + 2 * 4 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_floats_with_default_c(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        expected = 1.0 * 3.0 + 2.0 * 4.0 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_mixed_floats_and_ints_with_default_c(self):
        x = [1, 2]
        y = [3.0, 4.0]
        expected = 1 * 3.0 + 2 * 4.0 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_numpy_ints_with_default_c(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        expected = 1 * 3 + 2 * 4 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_numpy_floats_with_default_c(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        expected = 1.0 * 3.0 + 2.0 * 4.0 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_numpy_mixed_floats_and_ints_with_default_c(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        expected = 1 * 3.0 + 2 * 4.0 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_ints_with_c_equals_2(self):
        x = [1, 2]
        y = [3, 4]
        c = 2
        expected = (1 * 3 + 2 * 4) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_floats_with_c_equals_2(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        c = 2.0
        expected = (1 * 3 + 2 * 4) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y, c)}"

    def test_basic_mixed_floats_and_ints_with_c_equals_2(self):
        x = [1, 2]
        y = [3.0, 4.0]
        c = 2
        expected = (1 * 3.0 + 2 * 4.0) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y, c)}"

    def test_basic_numpy_ints_with_c_equals_2(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        c = 2
        expected = (1 * 3 + 2 * 4) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y, c)}"

    def test_basic_numpy_floats_with_c_equals_2(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        c = 2.0
        expected = (1.0 * 3.0 + 2.0 * 4.0) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y, c)}"

    def test_basic_numpy_mixed_floats_and_ints_with_c_equals_2(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        c = 2
        expected = (1 * 3.0 + 2 * 4.0) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_ints(self):
        x = [1, 2]
        y = [3, 4]
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_floats(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_mixed(self):
        x = [1, 2]
        y = [3.0, 4.0]
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_numpy_ints(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_numpy_floats(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_numpy_mixed(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"


class TestQuadraticKernel:
    def test_basic_ints(self):
        x = [1, 2]
        y = [3, 4]
        default_c: float = 1.0
        expected = ((1 * 3 + 2 * 4) + default_c) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_floats(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        default_c: float = 1.0
        expected = ((1.0 * 3.0 + 2.0 * 4.0) + default_c) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_mixed_floats_and_ints(self):
        x = [1, 2]
        y = [3.0, 4.0]
        default_c: float = 1.0
        expected = ((1 * 3.0 + 2 * 4.0) + default_c) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_numpy_ints(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        default_c: float = 1.0
        expected = ((1 * 3 + 2 * 4) + default_c) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_numpy_floats(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        default_c: float = 1.0
        expected = ((1.0 * 3.0 + 2.0 * 4.0) + default_c) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_numpy_mixed_floats_and_ints(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        default_c: float = 1.0
        expected = ((1 * 3.0 + 2 * 4.0) + default_c) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"


class TestPolynomialKernel:
    def test_value_error_raised_for_non_integer_degree(self):
        x = [1, 2]
        y = [3, 4]
        degree = 3.0
        with pytest.raises(ValueError):
            polynomial_kernel(x, y, degree=degree)

    def test_basic_ints_with_default_degree_and_c(self):
        x = [1, 2]
        y = [3, 4]
        degree = 3
        c = 1.0
        expected = (1 * 3 + 2 * 4 + c) ** degree
        assert (
            polynomial_kernel(x, y) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y)}"

    def test_basic_floats_with_default_degree_and_c(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        degree = 3
        c = 1.0
        expected = (1.0 * 3.0 + 2.0 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y)}"

    def test_basic_mixed_floats_and_ints_with_default_degree_and_c(self):
        x = [1, 2]
        y = [3.0, 4.0]
        degree = 3
        c = 1.0
        expected = (1 * 3.0 + 2 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y)}"

    def test_basic_numpy_ints_with_default_degree_and_c(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        degree = 3
        c = 1.0
        expected = (1 * 3 + 2 * 4 + c) ** degree
        assert (
            polynomial_kernel(x, y) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y)}"

    def test_basic_numpy_floats_with_default_degree_and_c(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        degree = 3
        c = 1.0
        expected = (1.0 * 3.0 + 2.0 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y)}"

    def test_basic_numpy_mixed_floats_and_ints_with_default_degree_and_c(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        degree = 3
        c = 1.0
        expected = (1 * 3.0 + 2 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y)}"

    def test_basic_ints_with_degree_4_and_c_2(self):
        x = [1, 2]
        y = [3, 4]
        degree = 4
        c = 2
        expected = (1 * 3 + 2 * 4 + c) ** degree
        assert (
            polynomial_kernel(x, y, degree, c) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_basic_floats_with_degree_4_and_c_2(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        degree = 4
        c = 2.0
        expected = (1.0 * 3.0 + 2.0 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y, degree, c) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_basic_mixed_floats_and_ints_with_degree_4_and_c_2(self):
        x = [1, 2]
        y = [3.0, 4.0]
        degree = 4
        c = 2.0
        expected = (1 * 3.0 + 2 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y, degree, c) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_basic_numpy_ints_with_degree_4_and_c_2(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        degree = 4
        c = 2
        expected = (1 * 3 + 2 * 4 + c) ** degree
        assert (
            polynomial_kernel(x, y, degree, c) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_basic_numpy_floats_with_degree_4_and_c_2(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        degree = 4
        c = 2.0
        expected = (1.0 * 3.0 + 2.0 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y, degree, c) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_basic_numpy_mixed_floats_and_ints_with_degree_4_and_c_2(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        degree = 4
        c = 2.0
        expected = (1 * 3.0 + 2 * 4.0 + c) ** degree
        assert (
            polynomial_kernel(x, y, degree, c) == expected
        ), f"Expected {expected}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_polynomial_kernel_with_degree_2_and_c_1_is_quadratic_kernel_with_c_1_ints(
        self,
    ):
        x = [1, 2]
        y = [3, 4]
        degree = 2
        c = 1
        assert polynomial_kernel(x, y, degree, c) == quadratic_kernel(
            x,
            y,
            c,
        ), f"Expected {quadratic_kernel(x, y, c)}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_polynomial_kernel_with_degree_2_and_c_1_is_quadratic_kernel_with_c_1_floats(
        self,
    ):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        degree = 2
        c = 1.0
        assert polynomial_kernel(x, y, degree, c) == quadratic_kernel(
            x,
            y,
            c,
        ), f"Expected {quadratic_kernel(x, y, c)}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_polynomial_kernel_with_degree_2_and_c_1_is_quadratic_kernel_with_c_1_mixed(
        self,
    ):
        x = [1, 2]
        y = [3.0, 4.0]
        degree = 2
        c = 1.0
        assert polynomial_kernel(x, y, degree, c) == quadratic_kernel(
            x,
            y,
            c,
        ), f"Expected {quadratic_kernel(x, y, c)}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_polynomial_kernel_with_degree_2_and_c_1_is_quadratic_kernel_with_c_1_numpy_ints(
        self,
    ):
        x = np.array([1, 2])
        y = np.array([3, 4])
        degree = 2
        c = 1
        assert polynomial_kernel(x, y, degree, c) == quadratic_kernel(
            x,
            y,
            c,
        ), f"Expected {quadratic_kernel(x, y, c)}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_polynomial_kernel_with_degree_2_and_c_1_is_quadratic_kernel_with_c_1_numpy_floats(
        self,
    ):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        degree = 2
        c = 1.0
        assert polynomial_kernel(x, y, degree, c) == quadratic_kernel(
            x,
            y,
            c,
        ), f"Expected {quadratic_kernel(x, y, c)}, but got {polynomial_kernel(x, y, degree, c)}"

    def test_polynomial_kernel_with_degree_2_and_c_1_is_quadratic_kernel_with_c_1_numpy_mixed(
        self,
    ):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        degree = 2
        c = 1.0
        assert polynomial_kernel(x, y, degree, c) == quadratic_kernel(
            x,
            y,
            c,
        ), f"Expected {quadratic_kernel(x, y, c)}, but got {polynomial_kernel(x, y, degree, c)}"


class TestRBFGaussianKernel:
    def test_rbf_gaussian_kernel_with_sigma_0_raises_value_error(self):
        x = [1, 2]
        y = [3, 4]
        sigma = 0
        with pytest.raises(ValueError):
            rbf_gaussian_kernel(x, y, sigma)

    def test_rbf_gaussian_kernel_works_without_np_linalg_norm_call(self):
        x = [1, 2]
        y = [3, 4]
        sigma = 1.0
        # Compute the expected value using np.linalg.norm
        expected = np.exp(
            -np.linalg.norm(np.array(x) - np.array(y)) ** 2 / (2 * sigma**2),
        )

        # Compare with the result from rbf_gaussian_kernel
        result = rbf_gaussian_kernel(x, y, sigma)
        assert np.isclose(
            result,
            expected,
        ), f"rbf_gaussian_kernel without np.linalg.norm is not working as expected. Expected {expected}, but got {result}"

    def test_rbf_gaussian_kernel_with_default_sigma_ints(self):
        x = [1, 2]
        y = [3, 4]
        sigma = 1.0
        x_y_diff = np.array(x) - np.array(y)
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y)}"

    def test_rbf_gaussian_kernel_with_default_sigma_floats(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        sigma = 1.0
        x_y_diff = np.array(x) - np.array(y)
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y)}"

    def test_rbf_gaussian_kernel_with_default_sigma_mixed_floats_and_ints(self):
        x = [1, 2]
        y = [3.0, 4.0]
        sigma = 1.0
        x_y_diff = np.array(x) - np.array(y)
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y)}"

    def test_rbf_gaussian_kernel_with_default_sigma_numpy_ints(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        sigma = 1.0
        x_y_diff = x - y
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y)}"

    def test_rbf_gaussian_kernel_with_default_sigma_numpy_floats(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        sigma = 1.0
        x_y_diff = x - y
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y)}"

    def test_rbf_gaussian_kernel_with_default_sigma_numpy_mixed_floats_and_ints(self):
        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        sigma = 1.0
        x_y_diff = x - y
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y)}"

    @pytest.mark.parametrize(
        "x, y, sigma",
        [
            ([1, 2], [3, 4], 2.0),
            ([1.0, 2.0], [3.0, 4.0], 2.0),
            ([1, 2], [3.0, 4.0], 2.0),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_rbf_gaussian_kernel_with_sigma_2(self, x, y, sigma):
        x_y_diff = np.array(x) - np.array(y)
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y, sigma) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y, sigma)}"

    @pytest.mark.parametrize(
        "x, y, sigma",
        [
            (np.array([1, 2]), np.array([3, 4]), 2.0),
            (np.array([1.0, 2.0]), np.array([3.0, 4.0]), 2.0),
            (np.array([1, 2]), np.array([3.0, 4.0]), 2.0),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_rbf_gaussian_kernel_with_sigma_2_numpy(self, x, y, sigma):
        x_y_diff = x - y
        x_y_diff_squared = x_y_diff**2
        sum_x_y_diff_squared = np.sum(x_y_diff_squared)
        expected = np.exp(-sum_x_y_diff_squared / (2 * sigma**2))
        assert (
            rbf_gaussian_kernel(x, y, sigma) == expected
        ), f"Expected {expected}, but got {rbf_gaussian_kernel(x, y, sigma)}"


class TestExponentialKernel:
    def test_exponential_kernel_with_sigma_0_raises_value_error(self):
        x = [1, 2]
        y = [3, 4]
        sigma = 0
        with pytest.raises(ValueError):
            exponential_kernel(x, y, sigma)

    @pytest.mark.parametrize(
        "x, y",
        [
            ([1, 2], [3, 4]),
            ([1.0, 2.0], [3.0, 4.0]),
            ([1, 2], [3.0, 4.0]),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_exponential_kernel_with_default_sigma(self, x, y):
        sigma = 1.0
        squared_xy0_diff = (x[0] - y[0]) ** 2
        squared_xy1_diff = (x[1] - y[1]) ** 2
        root_sum_squared_diff = np.sqrt(squared_xy0_diff + squared_xy1_diff)
        expected = np.exp(-root_sum_squared_diff / sigma)
        test_result = exponential_kernel(x, y)
        assert np.isclose(
            test_result,
            expected,
        ), f"Expected {expected}, but got {test_result}"

    @pytest.mark.parametrize(
        "x, y, sigma",
        [
            ([1, 2], [3, 4], 2.0),
            ([1.0, 2.0], [3.0, 4.0], 2.0),
            ([1, 2], [3.0, 4.0], 2.0),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_exponential_kernel_with_sigma_2(self, x, y, sigma):
        test_sigma = sigma
        squared_xy0_diff = (x[0] - y[0]) ** 2
        squared_xy1_diff = (x[1] - y[1]) ** 2
        root_sum_squared_diff = np.sqrt(squared_xy0_diff + squared_xy1_diff)
        expected = np.exp(-root_sum_squared_diff / sigma)
        test_result = exponential_kernel(x, y, test_sigma)
        assert np.isclose(
            test_result,
            expected,
        ), f"Expected {expected}, but got {test_result}"

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([1, 2]), np.array([3, 4])),
            (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
            (np.array([1, 2]), np.array([3.0, 4.0])),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_exponential_kernel_with_default_sigma_numpy(self, x, y):
        sigma = 1.0
        squared_xy_diff = (x - y) ** 2
        sum_squared_diff = np.sum(squared_xy_diff)
        root_sum_squared_diff = np.sqrt(sum_squared_diff)
        expected = np.exp(-root_sum_squared_diff / sigma)
        test_result = exponential_kernel(x, y)
        assert np.isclose(
            test_result,
            expected,
        ), f"Expected {expected}, but got {test_result}"

    @pytest.mark.parametrize(
        "x, y, sigma",
        [
            (np.array([1, 2]), np.array([3, 4]), 2.0),
            (np.array([1.0, 2.0]), np.array([3.0, 4.0]), 2.0),
            (np.array([1, 2]), np.array([3.0, 4.0]), 2.0),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_exponential_kernel_with_sigma_2_numpy(self, x, y, sigma):
        test_sigma = sigma
        squared_xy_diff = (x - y) ** 2
        sum_squared_diff = np.sum(squared_xy_diff)
        root_sum_squared_diff = np.sqrt(sum_squared_diff)
        expected = np.exp(-root_sum_squared_diff / sigma)
        test_result = exponential_kernel(x, y, test_sigma)
        assert np.isclose(
            test_result,
            expected,
        ), f"Expected {expected}, but got {test_result}"


class TestLaplacianKernel:
    @pytest.mark.parametrize(
        "gamma",
        [-1.0, 0.0],
        ids=["negative", "zero"],
    )
    def test_laplacian_kernel_with_gamma_leq_0_raises_value_error(self, gamma):
        x = [1, 2]
        y = [3, 4]
        test_gamma = gamma
        with pytest.raises(ValueError):
            laplacian_kernel(x, y, test_gamma)

    @pytest.mark.parametrize(
        "x, y, gamma",
        [
            ([1, 2], [3, 4], 2.0),
            ([1.0, 2.0], [3.0, 4.0], 2.0),
            ([1, 2], [3.0, 4.0], 2.0),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_laplacian_kernel_with_gamma_2(self, x, y, gamma):
        abs_xy0_diff = np.abs((x[0] - y[0]))
        abs_xy1_diff = np.abs((x[1] - y[1]))
        summed_abs_diff = abs_xy0_diff + abs_xy1_diff
        expected = np.exp(-gamma * summed_abs_diff)
        test_result = laplacian_kernel(x, y, gamma)
        assert np.isclose(
            test_result,
            expected,
        ), f"Expected {expected}, but got {test_result}"

    @pytest.mark.parametrize(
        "x, y, gamma",
        [
            (np.array([1, 2]), np.array([3, 4]), 2.0),
            (np.array([1.0, 2.0]), np.array([3.0, 4.0]), 2.0),
            (np.array([1, 2]), np.array([3.0, 4.0]), 2.0),
        ],
        ids=["ints", "floats", "mixed"],
    )
    def test_laplacian_kernel_with_gamma_2_numpy(self, x, y, gamma):
        abs_xy_diff = np.abs(x - y)
        summed_abs_diff = np.sum(abs_xy_diff)
        expected = np.exp(-gamma * summed_abs_diff)
        test_result = laplacian_kernel(x, y, gamma)
        assert np.isclose(
            test_result,
            expected,
        ), f"Expected {expected}, but got {test_result}"


def generate_id(param):
    kernel_func_name = ""
    kwargs_str = ""
    print(f"Debug param: {param} (type: {type(param)})")
    if callable(param):
        kernel_func_name = param.__name__
        return f"{kernel_func_name}"
    if isinstance(param, dict):
        kwargs_str = ", ".join(f"{k}={v}" for k, v in param.items())
        return f"with_{kwargs_str or 'no_args'}"


@pytest.mark.parametrize(
    "kernel_func, kwargs",
    [
        (linear_kernel, {}),
        (affine_kernel, {"c": 1.0}),
        (quadratic_kernel, {}),
        (polynomial_kernel, {"degree": 3, "c": 1.0}),
        (rbf_gaussian_kernel, {"sigma": 1.0}),
        (exponential_kernel, {"sigma": 1.0}),
        (laplacian_kernel, {"gamma": 1.0}),
    ],
    ids=generate_id,
)
def test_kernel_symmetry(kernel_func, kwargs):
    x = [1, 2]
    y = [3, 4]
    assert np.isclose(
        kernel_func(x, y, **kwargs),
        kernel_func(y, x, **kwargs),
    ), f"Expected {kernel_func(y, x, **kwargs)}, but got {kernel_func(x, y, **kwargs)}"
