import typing as t
from multiprocessing import Pool
from .types import Dataset, Association, DataSeries
from .associate import Associator
from .utils import make_combinations

#
# DEPRECATED
#
# We sadly did not have enough time to fully implement this, thus we reverted
# to already existing solutions. This could serve as a base for the future.
#

class DatasetAssociatorAdapter(t.Generic[Dataset]):
    """
    Adapts the Associator for use with pool.map
    """

    _associator: Associator

    def __init__(self, base_confidence: float):
        self._associator = Associator(base_confidence=base_confidence)

    def __call__(self, data_series: t.Tuple[DataSeries, DataSeries]):
        data_series_a, data_series_b = data_series
        return self._associator.associate(
            data_series_a=data_series_a, data_series_b=data_series_b
        )


class DatasetAssociator(t.Generic[Dataset]):
    """
    Performs association
    """

    _processing_pool_size: int

    def __init__(self, processing_pool_size: int):
        if processing_pool_size < 0:
            processing_pool_size = 1

        self._processing_pool_size = processing_pool_size

    def run(self, data: t.List[Dataset], base_confidence: float) -> t.List[Association]:
        """
        Run the associator
        """
        data_series_key_combinations = make_combinations(
            attrs_list=[key for key in dir(data[0]) if not key.startswith("_")], k=2
        )
        data_series_combinations = [
            (
                DataSeries(key=key_a, values=[getattr(entry, key_a) for entry in data]),
                DataSeries(key=key_b, values=[getattr(entry, key_b) for entry in data]),
            )
            for key_a, key_b in data_series_key_combinations
        ]
        results: t.List[t.Optional[Association]] = []

        with Pool(self._processing_pool_size) as pool:
            results = pool.map(
                DatasetAssociatorAdapter(base_confidence=base_confidence),
                data_series_combinations,
            )

        print(results)

        results = [result for result in results if result is not None]
        return results
