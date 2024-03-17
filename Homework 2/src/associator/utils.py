import itertools
import typing as t


def make_combinations(attrs_list: t.List[str], k: int = 2):
    return itertools.combinations(attrs_list, k)
