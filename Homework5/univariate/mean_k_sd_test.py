from mean_k_sd import get_outliers_mean_k_sd


def test_data_example():
    data = [2, 1, 2, 4, 62, 1]
    outliers = get_outliers_mean_k_sd(data=data, k = 2)

    assert outliers.size == 1
    assert outliers[0] == 62

def test_data_example_2():
    data = [2, 1, 2, 4, 62, 1]
    outliers = get_outliers_mean_k_sd(data=data, k = 3)

    assert outliers.size == 0
