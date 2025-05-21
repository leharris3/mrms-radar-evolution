import torch
import pickle
import pandas as pd
import xarray as xr
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List
from torch.utils.data.dataset import Dataset
from concurrent.futures import ThreadPoolExecutor

BASEDIR = "data/MergedReflectivityComposite"
EVENTS_DIR = "data/2024-tornado-MergedReflectivityComposite-0.5km/events"
CACHE_DIR = "data/2024-tornado-MergedReflectivityComposite-0.5km/__cache__"
METADATA = "data/2024-tornado-MergedReflectivityComposite-0.5km/metadata/tornados.csv"

TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15


class TornadoReflectivityDataset(Dataset):
    """
    Dataset of 2024 tornado events with MRMS 0.5km reflectivity data.
    """

    def __init__(self, split: str = "train"):

        super().__init__()
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split

        self.metadata = pd.read_csv(METADATA)
        self.train_pairs: List[np.ndarray] = []
        self.val_pairs: List[np.ndarray] = []

        # global max/min cell vals for normalization
        self.min_val = None
        self.max_val = None

        self._load_data()
        self._get_dataset_stats()

    def _get_tornado_data_pair(self, row) -> Tuple[str, str]:

        yearmonth = str(row.BEGIN_YEARMONTH)
        day = int(row.BEGIN_DAY)
        day = f"{day:02d}"

        subdir_name = yearmonth + day

        # reflectivity_dir = Path(BASEDIR) / subdir_name
        start_time = row.BEGIN_TIME
        end_time = row.END_TIME
        scale = row.TOR_F_SCALE

        event_name = "_".join([subdir_name, str(start_time), str(end_time), scale])

        # path to events dir
        events_dir = Path(EVENTS_DIR) / event_name

        data_paths = sorted([str(f) for f in events_dir.glob("*.grib2")])
        return data_paths[0], data_paths[1]

    def _load_data(
        self,
    ):

        buffer = []
        for _, row in tqdm(
            self.metadata.iterrows(),
            total=len(self.metadata),
            desc="🌪️ Loading tornado events from cache:",
        ):

            try:

                source_fp, target_fp = self._get_tornado_data_pair(row)
                event_name = source_fp.split("/")[3]

                # if cached, load from cache
                if Path(f"{CACHE_DIR}/{event_name}.pkl").exists():
                    with open(f"{CACHE_DIR}/{event_name}.pkl", "rb") as f:

                        source_arr, target_arr = pickle.load(f)

                        # NOTE: clip arrays to [0, inf]
                        source_arr = np.clip(source_arr, 0, None)
                        target_arr = np.clip(target_arr, 0, None)
                        
                        buffer.append([source_arr, target_arr])
                else:
                    print(f"Error: {event_name} could not be loaded from cache...")

                # source_data = xr.open_dataset(source_fp)
                # target_data = xr.open_dataset(target_fp)

                # lat_min, lat_max, lon_min, lon_max = (
                #     row.BEGIN_LAT,
                #     row.END_LAT,
                #     row.BEGIN_LON,
                #     row.END_LON,
                # )

                # # convert -180-180 -> 0-360
                # lon_min, lon_max = (((lon_min + 360) % 360)), (((lon_max + 360) % 360))

                # # a bit hacky
                # # -> [224, 224]
                # current_gap = lat_max - lat_min
                # TARGET_GAP = 2.24
                # BUFFER = (TARGET_GAP - current_gap) / 2

                # lat_min -= BUFFER
                # lat_max += BUFFER
                # lon_min -= BUFFER
                # lon_max += BUFFER

                # lat_mask = (source_data.latitude >= lat_min) & (
                #     source_data.latitude <= lat_max
                # )
                # lon_mask = (source_data.longitude >= lon_min) & (
                #     source_data.longitude <= lon_max
                # )

                # # crop -> [224, 224]
                # source_data = source_data.where(lat_mask & lon_mask, drop=True)
                # target_data = target_data.where(lat_mask & lon_mask, drop=True)

                # source_arr = source_data.to_array().values.squeeze()
                # target_arr = target_data.to_array().values.squeeze()

                # # cache
                # with open(f"{CACHE_DIR}/{event_name}.pkl", "wb") as f:
                #     pickle.dump([source_arr, target_arr], f)
                # buffer.append([source_arr, target_arr])

            except:

                pass

        # assign train/val splits
        self.train_pairs = buffer[: int(len(buffer) * TRAIN_SPLIT)]
        self.val_pairs = buffer[int(len(buffer) * TRAIN_SPLIT) :]

    def _get_dataset_stats(self):

        # calculate minimum and maximum values
        self.min_val = float("inf")
        self.max_val = float("-inf")

        for source_arr, target_arr in tqdm(self.train_pairs):
            self.min_val = min(self.min_val, source_arr.min())
            self.max_val = max(self.max_val, source_arr.max())

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.train_pairs)
        else:
            return len(self.val_pairs)

    def __getitem__(self, idx) -> dict:

        if self.split == "train":
            source_arr, target_arr = self.train_pairs[idx]
        else:
            source_arr, target_arr = self.val_pairs[idx]

        # convert to tensors
        source_arr = torch.tensor(source_arr, dtype=torch.float32)
        target_arr = torch.tensor(target_arr, dtype=torch.float32)

        # scale -> [0, 1]
        source_arr = (source_arr - self.min_val) / (self.max_val - self.min_val)
        target_arr = (target_arr - self.min_val) / (self.max_val - self.min_val)

        return {
            "source": source_arr,
            "target": target_arr,
        }


if __name__ == "__main__":
    dataset = TornadoReflectivityDataset(split="train")
    print(len(dataset))
    print(dataset[0])
