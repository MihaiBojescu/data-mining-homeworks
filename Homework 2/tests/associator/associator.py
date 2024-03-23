import unittest
from dataclasses import dataclass
from src.associator.associator import DatasetAssociator


@dataclass
class TestPerson:
    name: str
    age: str
    weight: int
    height: float


class DatasetAssociatorTests(unittest.TestCase):
    def test_should_filter_results(self):
        data = [
            TestPerson(name="Bob", age=24, weight=80, height=1.80),
            TestPerson(name="Alice", age=24, weight=65, height=1.70),
        ]

        sut = DatasetAssociator(2)
        result = sut.run(data=data, base_confidence=0.99)

        self.assertTrue(len(result) == 0)

    def test_should_return_results(self):
        data = [
            TestPerson(name="Bob", age=24, weight=80, height=1.80),
            TestPerson(name="Alice", age=24, weight=65, height=1.70),
        ]

        sut = DatasetAssociator(2)
        result = sut.run(data=data, base_confidence=0.99)

        self.assertTrue(result)
