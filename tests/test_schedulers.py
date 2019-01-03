import unittest
from modules import schedulers

class TestSchedulers(unittest.TestCase):

    def test_triangular_f(self):
        actual = schedulers._triangular_f(0, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = schedulers._triangular_f(200, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = schedulers._triangular_f(100, 100, 0, 1)
        expected = 1
        self.assertEqual(actual, expected)

        actual = schedulers._triangular_f(50, 100, 0, 1)
        expected = 0.5
        self.assertEqual(actual, expected)

    def test_triangular2_f(self):
        actual = schedulers._triangular2_f(0, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = schedulers._triangular2_f(200, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = schedulers._triangular2_f(300, 100, 0, 1)
        expected = 0.5
        self.assertEqual(actual, expected)

        actual = schedulers._triangular2_f(500, 100, 0, 1)
        expected = 0.25
        self.assertEqual(actual, expected)

        actual = schedulers._triangular2_f(700, 100, 0, 1)
        expected = 0.125
        self.assertEqual(actual, expected)

    def test_decay_f(self):
        actual = schedulers._decay_f(0, 100, 0, 1)
        expected = 1
        self.assertEqual(actual, expected)

        actual = schedulers._decay_f(25, 100, 0, 1)
        expected = 0.75
        self.assertEqual(actual, expected)

        actual = schedulers._decay_f(50, 100, 0, 1)
        expected = 0.5
        self.assertEqual(actual, expected)

        actual = schedulers._decay_f(100, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = schedulers._decay_f(150, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = schedulers._decay_f(24601, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

    def test_exp_f(self):
        actual = schedulers._exp_f(0, 0.95, 1)
        expected = 1
        self.assertEqual(actual, expected)

        actual = schedulers._exp_f(1, 0.95, 1)
        expected = 0.95
        self.assertEqual(actual, expected)

        actual = schedulers._exp_f(5, 0.1, 1)
        expected = 1e-5
        self.assertEqual(round(actual, 10), round(expected, 10))

    def test_exp_range_f(self):
        actual = schedulers._exp_range_f(0, 0.95, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = schedulers._exp_range_f(200, 0.95, 100, 1, 10)
        expected = (0.95 ** 200)
        self.assertEqual(round(actual, 10), round(expected, 10))

        actual = schedulers._exp_range_f(100, 0.95, 100, 1, 10)
        expected = 10 * (0.95 ** 100)
        self.assertEqual(round(actual, 10), round(expected, 10))

