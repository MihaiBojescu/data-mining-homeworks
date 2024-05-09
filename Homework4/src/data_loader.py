import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import typing as t
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


PARENT_DIR = str(os.path.join(Path(__file__).parent.absolute(), '..'))


def load_dataset(path: str, delimiter: str) -> tuple[np.array, np.array]:
    with open(path, "r") as fd:
        dataset = pd.read_csv(fd, delimiter=delimiter).to_numpy()
        shape = dataset.shape
        assert shape[1] > 2
        return dataset[:, :-1], dataset[:, -1:]


def dataset_2d_10c() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(PARENT_DIR, "./data/2d-10c.dat"), " ")


def dataset_iris() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(PARENT_DIR, "./data/iris.csv"), ",")


def dataset_long() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(PARENT_DIR, "./data/long.dat"), " ")


def dataset_order2_3clust() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(PARENT_DIR, "./data/order2-3clust.csv"), ",")


def dataset_smile() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(PARENT_DIR, "./data/smile.csv"), ",")


def dataset_square() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(PARENT_DIR, "./data/square.dat"), " ")


# Obesity Dataset

LABEL_VARIABLE = "NObeyesdad"
NUMERICAL_VARIABLES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL_VARIABLES_NO_LABEL = [
    "FAVC",
    "CAEC",
    "CALC",
    "SCC",
    "MTRANS",
    "Gender",
    "family_history_with_overweight",
    "SMOKE",
]
CATEGORICAL_VARIABLES = [
    *CATEGORICAL_VARIABLES_NO_LABEL,
    LABEL_VARIABLE,
]
ALL_VARIABLES_NO_LABEL = [*NUMERICAL_VARIABLES, *CATEGORICAL_VARIABLES_NO_LABEL]
ALL_VARIABLES = [*NUMERICAL_VARIABLES, *CATEGORICAL_VARIABLES]
LABEL_DICTIONARY = {
    "Age": "Age",
    "Height": "Height (cm)",
    "Weight": "Weight (kg)",
    "FCVC": " Frequency of consumption of vegetables (times per day)",
    "NCP": "Number of main meals",
    "CH2O": "Consumption of water daily (Liters)",
    "FAF": "Physical activity frequency (times per day)",
    "TUE": "Time using technology devices (hours)",
    "FAVC": "Frequent consumption of high caloric food",
    "CAEC": "Consumption of food between meals",
    "CALC": "Consumption of alcohol",
    "SCC": "Calories consumption monitoring",
    "MTRANS": "Transportation used",
    "Gender": "Gender",
    "family_history_with_overweight": "Family member suffered or suffers from overweight",
    "SMOKE": "Smoker or not",
    "NObeyesdad": "Obesity level",
}

T = t.TypeVar("T")


class Person:
    Gender: str
    Age: np.int32
    Height: np.float32
    Weight: np.float32
    family_history_with_overweight: str
    FAVC: str
    FCVC: np.float32
    NCP: np.float32
    CAEC: str
    SMOKE: str
    CH2O: np.float32
    SCC: str
    FAF: np.float32
    TUE: np.float32
    CALC: str
    MTRANS: str
    NObeyesdad: str

    def __init__(
        self,
        Gender: str,
        Age: str,
        Height: str,
        Weight: str,
        family_history_with_overweight: str,
        FAVC: str,
        FCVC: str,
        NCP: str,
        CAEC: str,
        SMOKE: str,
        CH2O: str,
        SCC: str,
        FAF: str,
        TUE: str,
        CALC: str,
        MTRANS: str,
        NObeyesdad: str,
    ):
        self.Gender = Gender
        self.Age = np.float32(Age)
        self.Height = np.float32(Height)
        self.Weight = np.float32(Weight)
        self.family_history_with_overweight = family_history_with_overweight
        self.FAVC = FAVC
        self.FCVC = np.float32(FCVC)
        self.NCP = np.float32(NCP)
        self.CAEC = CAEC
        self.SMOKE = SMOKE
        self.CH2O = np.float32(CH2O)
        self.SCC = SCC
        self.FAF = np.float32(FAF)
        self.TUE = np.float32(TUE)
        self.CALC = CALC
        self.MTRANS = MTRANS
        self.NObeyesdad = NObeyesdad

    def __str__(self):
        return vars(self)

    def __len__(self):
        return len(vars(self))

    def __repr__(self):
        return vars(self)


class DatasetManager:
    def __init__(self, path_to_csv: str):
        self.path_to_csv = path_to_csv

    def load_as_obj_list(self) -> list[Person]:
        with open(self.path_to_csv) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            return [Person(**row) for row in csv_reader]

    @staticmethod
    def process_dataframe_one_hot(
        arg_dataset: pd.DataFrame,
        feature_names: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        result_features_dataframe = pd.DataFrame()

        feature_names = set(feature_names)

        for variable in set(NUMERICAL_VARIABLES).intersection(feature_names):
            result_features_dataframe = pd.concat(
                [result_features_dataframe, arg_dataset[variable]], axis=1
            )

        for variable in set(CATEGORICAL_VARIABLES_NO_LABEL).intersection(feature_names):
            result_with_dummies = pd.get_dummies(arg_dataset[variable]).astype(float)
            result_features_dataframe = pd.concat(
                [result_features_dataframe, result_with_dummies], axis=1
            )

        pd.get_dummies(arg_dataset[LABEL_VARIABLE])

        result_labels_dataframe = pd.DataFrame(
            {
                LABEL_VARIABLE: LabelEncoder()
                .fit_transform(arg_dataset[LABEL_VARIABLE])
                .tolist()
            }
        )

        return result_features_dataframe, result_labels_dataframe


def dataset_obesity() -> tuple[np.array, np.array]:
    dataset_manager = DatasetManager(os.path.join(PARENT_DIR, "./data/ObesityDataSet.csv"))
    dataset = pd.DataFrame.from_records(data=[vars(entry) for entry in dataset_manager.load_as_obj_list()])

    #features, labels = DatasetManager.process_dataframe_one_hot(dataset, ["Height", "Weight", "Age", "Gender", "family_history_with_overweight"])
    features_dataset, labels = DatasetManager.process_dataframe_one_hot(dataset, ["Height", "Weight"])

    height_weight_dataset = features_dataset[["Height", "Weight"]]
    adjusted = {
        "AdjustedHeight": [],
        "AdjustedWeight": []
    }
    origin = min(height_weight_dataset["Height"].tolist()), min(height_weight_dataset["Weight"].tolist())

    for it in range(len(height_weight_dataset)):
        height, weight = height_weight_dataset["Height"][it], height_weight_dataset["Weight"][it]

        adjusted_height, adjusted_weight = rotate_matrix(origin, (height, weight), 0.01)

        adjusted["AdjustedHeight"].append(adjusted_height)
        adjusted["AdjustedWeight"].append(adjusted_weight)

    adjusted["AdjustedWeight"] = MinMaxScaler(feature_range=(0, 0.1)).fit_transform(
        np.array(adjusted["AdjustedWeight"]).reshape(-1, 1)).reshape(-1).tolist()

    adjusted["AdjustedHeight"] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.array(adjusted["AdjustedHeight"]).reshape(-1, 1)).reshape(-1).tolist()

    adjusted_dataframe = pd.DataFrame(adjusted)
    adjusted_dataframe.reset_index(drop=True, inplace=True)

    # supplemental_features, _ = DatasetManager.process_dataframe_one_hot(dataset, ["family_history_with_overweight"])
    # adjusted_dataframe = pd.concat([adjusted_dataframe, supplemental_features], axis=1)

    return adjusted_dataframe.to_numpy(), labels.to_numpy()


def rotate_matrix(origin: tuple[float, float], point: tuple[float, float], angle: float):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
