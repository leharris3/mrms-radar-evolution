import s3fs
import torch
import datetime
import pickle
import gzip
import shutil
import pandas as pd
import xarray as xr
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List
from torch.utils.data.dataset import Dataset
from concurrent.futures import ThreadPoolExecutor

PRODUCT = "MergedBaseReflectivity_00.50"
BASE_PRODUCT_URL = f"noaa-mrms-pds/CONUS/{PRODUCT}/"

# TODO: remove
BASEDIR = "data/MergedReflectivityComposite"

EVENTS_DIR = "data/2024-tornado-MergedReflectivityComposite-0.5km/events"
CACHE_DIR = "data/2024-tornado-MergedReflectivityComposite-0.5km/__cache__"
METADATA = "data/2024-tornado-MergedReflectivityComposite-0.5km/metadata/tornados.csv"

TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15


class RadarFileRetriever:
    """Class to retrieve radar file paths around a given event time."""

    BASE_PRODUCT_URL = f"noaa-mrms-pds/CONUS/{PRODUCT}/"

    def __init__(self):
        self.fs = s3fs.S3FileSystem(anon=True)

    def _get_datetime_from_filename(self, filename: str) -> datetime.datetime:
        """Extract datetime from MRMS filename."""
        timestamp = filename.split("_")[-1].split(".")[0]
        return datetime.datetime.strptime(timestamp, "%Y%m%d-%H%M%S")

    def _list_files_for_day(self, date: datetime.date) -> List[str]:
        """List all files from S3 for a given date."""
        date_str = date.strftime("%Y%m%d")
        path = Path(self.BASE_PRODUCT_URL) / date_str
        return self.fs.ls(path)

    def get_surrounding_files(self, t_0: datetime.datetime) -> List[str]:
        """Get 15 radar files before and 3 files after the event time t_0."""
        prev_files, next_files = [], []

        # Collect files from current, previous, and next days
        files_today = self._list_files_for_day(t_0.date())
        files_yesterday = self._list_files_for_day(
            t_0.date() - datetime.timedelta(days=1)
        )
        files_tomorrow = self._list_files_for_day(
            t_0.date() + datetime.timedelta(days=1)
        )

        all_files = files_yesterday + files_today + files_tomorrow
        sorted_files = sorted(
            all_files, key=lambda f: self._get_datetime_from_filename(f)
        )

        # Find the position of t_0
        pos = None
        for idx, file in enumerate(sorted_files):
            file_time = self._get_datetime_from_filename(file)
            if file_time >= t_0:
                pos = idx
                break

        if pos is None:
            raise ValueError("Event time is beyond available file times.")

        # Select 15 files before, handling boundary cases
        start_idx = max(pos - 15, 0)
        prev_files = sorted_files[start_idx:pos]

        # Select 3 files after, handling boundary cases
        end_idx = min(pos + 3, len(sorted_files))
        next_files = sorted_files[pos:end_idx]

        return prev_files + next_files


class TornadoReflectivityDataset(Dataset):
    """
    Dataset of MRMS tornado events circa 2021-2024.
    
    Things to implement:
        1. Robust interface for MRMS AWS S3 bucket.
        2. Class for managing StormEvents data.
    """

    def __init__(self, split: str = "train"):

        super().__init__()
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split

        self.metadata = pd.read_csv(METADATA)

        # [[15x source, 3x target]] data pairs
        self.train_pairs: List[np.ndarray] = []
        self.val_pairs: List[np.ndarray] = []

        # global max/min cell vals for normalization
        self.min_val = None
        self.max_val = None

        self._load_data()
        # self._get_dataset_stats()

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

        # holds all data pairs
        buffer = []

        # s3 conntection
        retriever = RadarFileRetriever()

        for _, row in tqdm(
            self.metadata.iterrows(),
            total=len(self.metadata),
            desc="🌪️ Loading tornado events from cache",
        ):

            # HACK: skip a few missing rows
            try:
                source_fp, target_fp = self._get_tornado_data_pair(row)
            except Exception as e:
                print(f"Error: {e}")
                continue

            event_name = Path(source_fp).parts[3]

            # 1. 15 images prior to t=0
            # 2. 3 images after t=0

            # YYMMDD
            t_0_str = event_name.split("_")[0]
            t_0_str = t_0_str[0:4] + "-" + t_0_str[4:6] + "-" + t_0_str[6:8]
            t_0 = datetime.datetime.strptime(t_0_str, "%Y-%m-%d")

            # [18]
            try:
                radar_files = retriever.get_surrounding_files(t_0)
            except RuntimeError as exc:
                print("No radar keys for %s – %s", event_name, exc)
                continue

            lat_min, lat_max, lon_min, lon_max = (
                row.BEGIN_LAT,
                row.END_LAT,
                row.BEGIN_LON,
                row.END_LON,
            )

            # convert -180-180 -> 0-360
            lon_min, lon_max = (((lon_min + 360) % 360)), (((lon_max + 360) % 360))

            # ----- scale to 224, 224 -----

            # avoid some floating point errors
            TARGET_GAP = 2.2401

            current_gap = lat_max - lat_min
            BUFFER = (TARGET_GAP - current_gap) / 2
            lat_min -= BUFFER
            lat_max += BUFFER

            current_gap = lon_max - lon_min
            BUFFER = (TARGET_GAP - current_gap) / 2
            lon_min -= BUFFER
            lon_max += BUFFER

            # ----------------------------

            subbuffer = []

            for fp in radar_files:

                cache_fp = f"{CACHE_DIR}/{event_name}/{fp.split('/')[-1].strip('.grib2.gz')}.pkl"

                # if cached, load from cache
                if Path(cache_fp).exists():

                    with open(cache_fp, "rb") as f:
                        data_arr = pickle.load(f)
                        subbuffer.append(data_arr)

                    continue

                # ---- load & cache ----

                gz_path = Path(EVENTS_DIR) / event_name / fp.split("/")[-1]
                read_path = gz_path.with_suffix("")

                # if fp doesn't exist, download
                if not Path(gz_path).exists():

                    # download .gz file from S3
                    retriever.fs.get(fp, read_path.parent, recursive=True)

                    # unzip + save .gz unzipped file
                    with gzip.open(gz_path, "rb") as f_in:
                        with open(read_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                data = xr.open_dataset(read_path, engine="cfgrib")

                # [20, 60]
                lat_mask = (data.latitude >= lat_min) & (data.latitude <= lat_max)

                # [230, 300]
                lon_mask = (data.longitude >= lon_min) & (data.longitude <= lon_max)

                # crop -> [224, 224]
                data = data.where(lat_mask & lon_mask, drop=True)
                data_arr = data.to_array().values.squeeze()

                # create dir if needed
                cache_dir = Path(cache_fp).parent
                if not cache_dir.exists():
                    cache_dir.mkdir(parents=True, exist_ok=True)

                # save to cache
                with open(cache_fp, "wb") as f:
                    pickle.dump(data_arr, f)

                subbuffer.append(data_arr)

            # [[source], [target]]
            buffer.append([subbuffer[:15], subbuffer[15:]])

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
        """ """

        if self.split == "train":
            source_arr, target_arr = self.train_pairs[idx]
        else:
            source_arr, target_arr = self.val_pairs[idx]

        # convert to tensors
        source_arr = torch.tensor(source_arr, dtype=torch.float32)
        target_arr = torch.tensor(target_arr, dtype=torch.float32)

        # todo: should we clip to (0, inf)?

        # scale -> [0, 1]
        # source_arr = (source_arr - self.min_val) / (self.max_val - self.min_val)
        # target_arr = (target_arr - self.min_val) / (self.max_val - self.min_val)

        return {
            "source": source_arr,
            "target": target_arr,
        }


if __name__ == "__main__":
    dataset = TornadoReflectivityDataset(split="train")
    print(len(dataset))
    print(dataset[0])
