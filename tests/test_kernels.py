from src.kernels import affine_kernel, linear_kernel, quadratic_kernel


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
        import numpy as np

        x = np.array([1, 2])
        y = np.array([3, 4])
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"

    def test_basic_numpy_floats(self):
        import numpy as np

        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert (
            linear_kernel(x, y) == 11.0
        ), f"Expected 11.0, but got {linear_kernel(x, y)}"

    def test_basic_numpy_mixed_floats_and_ints(self):
        import numpy as np

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
        import numpy as np

        x = np.array([1, 2])
        y = np.array([3, 4])
        expected = 1 * 3 + 2 * 4 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_numpy_floats_with_default_c(self):
        import numpy as np

        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        expected = 1.0 * 3.0 + 2.0 * 4.0 + 1.0
        assert (
            affine_kernel(x, y) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y)}"

    def test_basic_numpy_mixed_floats_and_ints_with_default_c(self):
        import numpy as np

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
        import numpy as np

        x = np.array([1, 2])
        y = np.array([3, 4])
        c = 2
        expected = (1 * 3 + 2 * 4) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y, c)}"

    def test_basic_numpy_floats_with_c_equals_2(self):
        import numpy as np

        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        c = 2.0
        expected = (1.0 * 3.0 + 2.0 * 4.0) + c
        assert (
            affine_kernel(x, y, c) == expected
        ), f"Expected {expected}, but got {affine_kernel(x, y, c)}"

    def test_basic_numpy_mixed_floats_and_ints_with_c_equals_2(self):
        import numpy as np

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
        import numpy as np

        x = np.array([1, 2])
        y = np.array([3, 4])
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_numpy_floats(self):
        import numpy as np

        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        c = 0
        assert affine_kernel(x, y, c) == linear_kernel(
            x,
            y,
        ), f"Expected {linear_kernel(x, y)}, but got {affine_kernel(x, y, c)}"

    def test_affine_kernel_with_c_0_is_linear_kernel_numpy_mixed(self):
        import numpy as np

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
        expected = (1 * 3 + 2 * 4) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_floats(self):
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        expected = (1.0 * 3.0 + 2.0 * 4.0) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_mixed_floats_and_ints(self):
        x = [1, 2]
        y = [3.0, 4.0]
        expected = (1 * 3.0 + 2 * 4.0) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_numpy_ints(self):
        import numpy as np

        x = np.array([1, 2])
        y = np.array([3, 4])
        expected = (1 * 3 + 2 * 4) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_numpy_floats(self):
        import numpy as np

        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        expected = (1.0 * 3.0 + 2.0 * 4.0) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"

    def test_basic_numpy_mixed_floats_and_ints(self):
        import numpy as np

        x = np.array([1, 2])
        y = np.array([3.0, 4.0])
        expected = (1 * 3.0 + 2 * 4.0) ** 2
        assert (
            quadratic_kernel(x, y) == expected
        ), f"Expected {expected}, but got {quadratic_kernel(x, y)}"
