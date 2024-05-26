from Homework5.data_loader import dataset_wine
from Homework5.run.first_task import run_univariate


def main():
    features, labels = dataset_wine()

    run_univariate(features=features, labels=labels)


if __name__ == "__main__":
    main()
