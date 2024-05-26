import os.path
import typing as t
import pandas as pd
import numpy as np
import torch.nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PARENT_DIR = str(os.path.join(Path(__file__).parent.absolute()))


class NPArrDataset(Dataset):
    def __init__(self, data_arr: np.array):
        self.__data_arr = data_arr.astype(np.float32)

    def get_full(self) -> torch.Tensor:
        return torch.from_numpy(self.__data_arr)

    def __len__(self):
        return self.__data_arr.shape[0]

    def __getitem__(self, idx: int):
        return self.__data_arr[idx, :]


class AutoencoderOutlierDetector:
    def __init__(self):
        self.__model: AE = None

    def build(self, features: np.array, model_state_dict_path: str):
        dataset = NPArrDataset(features)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.__model = AE(features.shape[1])
        model_state_dict_full_path = self.__get_model_path(model_state_dict_path)

        if model_state_dict_full_path is not None:
            print("Loading Autoencoder from Disk")
            self.__model.load_state_dict(torch.load(model_state_dict_full_path))
            self.__model.eval()
        else:
            self.__train(dataloader, 150)
            torch.save(self.__model.state_dict(), os.path.join(PARENT_DIR, model_state_dict_path))

    def __get_model_path(self, model_state_dict_path: str) -> t.Optional[str]:
        model_state_dict_full_path = os.path.join(PARENT_DIR, model_state_dict_path)

        if not os.path.exists(model_state_dict_full_path):
            return None
        
        return model_state_dict_full_path

    def __train(self, dataloader: DataLoader, nr_of_epochs: int):
        tbar = tqdm(tuple(range(nr_of_epochs)))

        for epoch in tbar:
            total_loss, reconstruction_error = self.__train_step(dataloader, "cpu")
            tbar.set_postfix_str(f"Total Loss: {total_loss}, Reconstruction Error: {reconstruction_error}")

    def __train_step(self, dataloader: DataLoader, device: str) -> tuple[float, float]:
        total_loss = 0.0
        reconstruction_error = 0.0

        optimizer = torch.optim.Adam(params=self.__model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        for data in dataloader:
            data = data.to(device, non_blocking=True)

            output = self.__model(data)
            loss = criterion(output, data)
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            reconstruction_error += torch.sum(torch.absolute(data - output)).item()

        return total_loss, reconstruction_error

    def predict(self, features: np.array) -> pd.DataFrame:
        dataset = NPArrDataset(features)
        data = dataset.get_full().to("cpu", non_blocking=True)
        output = self.__model(data)
        return pd.DataFrame({"outlier_scores": torch.sum(torch.absolute(data - output), axis=1).detach().numpy().tolist()})


class AE(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 48),
            torch.nn.Tanh(),
            torch.nn.Linear(48, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 4),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 48),
            torch.nn.Tanh(),
            torch.nn.Linear(48, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
