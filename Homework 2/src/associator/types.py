import typing as t
from dataclasses import dataclass

Dataset = t.TypeVar("Dataset")
DataSeriesValue = t.TypeVar("DataSeriesValue", str, int, float)


@dataclass
class DataSeries(t.Generic[DataSeriesValue]):
    """
    A data series.
    """

    key: str
    values: t.List[DataSeriesValue]


@dataclass
class Association:
    """
    An association between class `source` and class `destination` with confidence `confidence`.
    """

    source: str
    destination: str
    confidence: float
