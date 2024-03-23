import typing as t
from .types import Association, DataSeries, DataSeriesValue


class Associator:
    _base_confidence: float

    def __init__(self, base_confidence: float):
        if not 0 < base_confidence < 1:
            raise RuntimeError(f"Base confidence not valid: 0 < {base_confidence} < 1")

        self._base_confidence = base_confidence

    def associate(
        self,
        data_series_a: DataSeries,
        data_series_b: DataSeries,
    ):
        """
        Checks wether data_series_a (1 data series) can be associated with data_Series_b (1 data series).
        If the calculated confidence exceeds a given base confidence (0.0 < base_confidence < 1.0),
        then we consider that the data series can be associated.
        Else, we cannot consider that the data series are associated
        """

        association_dict: t.Dict[DataSeriesValue, t.Dict[DataSeriesValue, int]] = {}
        total_tuples = len(data_series_a.values)
        max_confidence = 0.0

        for entry_a, entry_b in zip(data_series_a.values, data_series_b.values):
            if not entry_a in association_dict:
                association_dict[entry_a] = {}

            if not entry_b in association_dict[entry_a]:
                association_dict[entry_a][entry_b] = 0

            association_dict[entry_a][entry_b] += 1

        for source in association_dict.values():
            for destination in source.values():
                confidence = destination / total_tuples
                max_confidence = max(max_confidence, confidence)

        if max_confidence < self._base_confidence:
            return None

        return Association(
            source=data_series_a.key,
            destination=data_series_b.key,
            confidence=max_confidence,
        )
