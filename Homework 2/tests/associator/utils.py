import unittest
from src.associator.utils import make_combinations


class CombinationsTests(unittest.TestCase):
    def test_should_return_combinations(self):
        result = list(make_combinations(['a', 'b', 'c'], 2))
        self.assertTrue(len(result), 3)

    def test_should_return_combinations_when_k_is_equal_to_length(self):
        result = list(make_combinations(['a', 'b', 'c'], 3))
        self.assertTrue(len(result), 1)

    def test_should_return_combinations_when_k_is_1(self):
        result = list(make_combinations(['a', 'b', 'c'], 1))
        self.assertTrue(len(result), 1)

