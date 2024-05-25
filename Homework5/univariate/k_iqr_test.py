from k_iqr import get_outliers_k_iqr


def test_data_example():
    data = [2, 1, 2, 4, 62, 1]
    outliers = get_outliers_k_iqr(data=data, k_iqr = 0.5)

    assert outliers.size == 1
    assert outliers[0] == 62

def test_data_example_2():
    data = [2, 1, 2, 4, 62, 1]
    outliers = get_outliers_k_iqr(data=data, k_iqr = 1.5)

    assert outliers.size == 1
    assert outliers[0] == 62
