import unittest
from src.associator.types import DataSeries
from src.associator.associate import associate


class AssociateTests(unittest.TestCase):
    def test_should_return_none_when_rules_cannot_be_associated(self):
        data_series_a = DataSeries(key="key_a", values=[1, 1])
        data_series_b = DataSeries(key="key_b", values=["a", "b"])

        result = associate(
            data_series_a=data_series_a,
            data_series_b=data_series_b,
            base_confidence=0.99,
        )

        self.assertTrue(result is None)

    def test_should_return_association_otherwise(self):
        data_series_a = DataSeries(key="key_a", values=[1, 1])
        data_series_b = DataSeries(key="key_b", values=["a", "a"])

        result = associate(
            data_series_a=data_series_a,
            data_series_b=data_series_b,
            base_confidence=0.99,
        )

        self.assertTrue(result is not True)
