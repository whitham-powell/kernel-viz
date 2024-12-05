from src.kernels import linear_kernel


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
