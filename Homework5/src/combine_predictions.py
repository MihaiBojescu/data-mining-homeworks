import numpy as np


def combine_predictions(predictions: list[np.array], min_votes: int):
    assert len(predictions) > 0

    result = predictions[0].astype(int)

    for it in range(1, len(predictions)):
        result += predictions[it].astype(int)

    return (result >= min_votes).astype(float)
