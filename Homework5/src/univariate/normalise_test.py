from normalise import normalise


def test_data_example():
    data = [2, 1, 2, 4, 62, 1]
    result = normalise(data=data)

    assert result[0] == 0.027777777777777776
    assert result[1] == 0.013888888888888888
    assert result[2] == 0.027777777777777776
    assert result[3] == 0.05555555555555555
    assert result[4] == 0.8611111111111112
    assert result[5] == 0.013888888888888888
