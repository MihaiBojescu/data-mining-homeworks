import typing as t
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolClass
from associator.types import Dataset
from associator.associate import associate
from associator.utils import make_combinations


class Associator(t.Generic[Dataset]):
    _pool: PoolClass

    def __init__(self, processing_pool_size: int):
        if processing_pool_size < 0:
            processing_pool_size = 1

        self._pool = Pool(processing_pool_size)

    def run(self, data: t.List[Dataset], base_confidence: float):
        combinations = make_combinations(
            [key for key in data[0] if key.startswith("_")]
        )

        results = self._pool.map(
            lambda data_series_a, data_series_b: associate(
                data_series_a, data_series_b, base_confidence
            ),
            combinations,
        )
        results = [result for result in results if result != None]

        return results
