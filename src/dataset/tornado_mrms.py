from __future__ import annotations

import os
import sys
import torch
import pickle
import gzip
import shutil
import pandas as pd
import xarray as xr
import numpy as np

from glob import glob
from bisect import bisect_left
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Tuple, List
from torch.utils.data.dataset import Dataset
from concurrent.futures import ThreadPoolExecutor

from src.utils.mrms.mrms import MRMSAWSS3Client, MRMSURLs
from src.utils.mrms.files import ZippedGrib2File, Grib2File


CACHE_DIR = "data/2022-2024-tornado-MRMS/__cache__"
METADATA = "data/2022-2024-tornado-MRMS/metadata/tornados.csv"

TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15


class TornadoMRMSDataset(Dataset):
    """
    Dataset of MRMS tornado events circa 2022-2024.
    """

    _EVENTS_DIR = "data/2022-2024-tornado-MRMS/events"
    _ALL_DATA_CACHE_FP = "data/2022-2024-tornado-MRMS/__cache__/__all_data__/all_data.h5"

    __products__ = ["MergedBaseReflectivity_00.50"]

    def __init__(self, split: str = "train", build_dataset=False):

        super().__init__()
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split

        self.metadata = pd.read_csv(METADATA)

        # [[6, 224, 224]]
        self.train_data: List[np.ndarray] = []
        self.val_data: List[np.ndarray] = []

        # global max/min cell vals for normalization
        # NOTE: we clip all samples [0, inf)
        self.min_val = 0
        self.max_val = None

        if build_dataset == True:
            self._build_dataset()

        self._load_data_from_cache()
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

    def _build_dataset(self):

        client = MRMSAWSS3Client()

        # HACK: we'll use one product for now
        product = TornadoMRMSDataset.__products__[0]
        prod_path = f"{MRMSURLs.BASE_URL_CONUS}{product}"
        events = {}

        @lru_cache(maxsize=64)
        def _ls_day(yyyymmdd: str) -> List[str]:
            
            files = []
            try:
                files = client.ls(f"{prod_path}/{yyyymmdd}")
            except:
                print(f"Error: no files found for day: {yyyymmdd}")

            files.sort()
            return files

        def _parse_fp_datetime(fp: str) -> datetime:
            """
            Extract a `datetime` from an MRMS filename.

            Example filename:
                MRMS_MergedReflectivityQC_00.50_20201014-220000.grib2.gz
            """
            fname = fp.split("/")[-1]
            ymd, hms = fname.split("_")[-1].split("-")
            return datetime.strptime(ymd + hms[:6], "%Y%m%d%H%M%S")

        # # 1. list all tornado events
        # for row in tqdm(
        #     self.metadata.itertuples(index=False), total=len(self.metadata)
        # ):

        #     # YYYYMMDD
        #     ymd = f"{row.BEGIN_YEARMONTH}{row.BEGIN_DAY:02}"
        #     begin_dt = datetime.strptime(
        #         f"{ymd}{row.BEGIN_TIME:06}",
        #         "%Y%m%d%H%M%S",
        #     )

        #     candidate_files: List[str] = []
        #     for offset in (-1, 0, 1):
        #         day = (begin_dt + timedelta(days=offset)).strftime("%Y%m%d")
        #         candidate_files.extend(_ls_day(day))

        #     if not candidate_files:
        #         print(f"Error: skipping a row!")
        #         continue

        #     times: List[datetime] = [_parse_fp_datetime(fp) for fp in candidate_files]

        #     # find the closest radar timestep
        #     pos = bisect_left(times, begin_dt)
        #     nearest_idx_candidates = [i for i in (pos - 1, pos) if 0 <= i < len(times)]
        #     nearest_idx = min(
        #         nearest_idx_candidates,
        #         key=lambda i: abs(times[i] - begin_dt),
        #         default=None,
        #     )

        #     if nearest_idx is None:
        #         print(f"Error: no index found!")
        #         continue

        #     start_time = row.BEGIN_TIME
        #     end_time = row.END_TIME
        #     scale = row.TOR_F_SCALE
        #     event_name = "_".join([ymd, str(start_time), str(end_time), scale])

        #     # add file paths to events dict
        #     events[event_name] = candidate_files[nearest_idx - 3 : nearest_idx + 3]

        # 2. for each (product, event), download neighboring radar state .gz files
        # events_cache_fp = "/playpen/mufan/levi/tianlong-chen-lab/torp-v2/mrms-radar-evolution/data/2022-2024-tornado-MRMS/metadata/all_events.pkl"
        # with open(events_cache_fp, "rb") as f: 
        #     events: dict = pickle.load(f)

        # out_dir = Path("/playpen/mufan/levi/tianlong-chen-lab/torp-v2/mrms-radar-evolution/data/2022-2024-tornado-MRMS/events")
        # for k, v in tqdm(events.items()):
        #     for fp in v:
        #         fp = fp.replace("noaa-mrms-pds/CONUS", "s3://noaa-mrms-pds/CONUS")
        #         curr_out_dir = out_dir / k
        #         os.makedirs(str(curr_out_dir), exist_ok=True)
        #         curr_out_fp  = curr_out_dir / Path(fp).name
        #         if Path(curr_out_fp).is_file(): ConnectionTimeoutError()
        #         client.download(str(fp), str(curr_out_fp), recursive=False)
                
        # 3. for each (event, .gz file) open, crop, cache, and delete the original
        
        events_dir = "/playpen/mufan/levi/tianlong-chen-lab/torp-v2/mrms-radar-evolution/data/2022-2024-tornado-MRMS/events"
        metadata_fp = "/playpen/mufan/levi/tianlong-chen-lab/torp-v2/mrms-radar-evolution/data/2022-2024-tornado-MRMS/metadata/tornados.csv"
        df = pd.read_csv(metadata_fp)

        from glob import glob
        from src.utils.mrms.files import ZippedGrib2File, Grib2File

        # files = glob(events_dir + "/*/*.grib2")
        # for fp in tqdm(files):
        #     path = Path(fp)
        #     gf = Grib2File(fp)
        #     # gf.

        def crop_xarray_to_224_224(row, arr):

            # get raw lat/lon
            lat_min, lat_max, lon_min, lon_max = (
                row.BEGIN_LAT,
                row.END_LAT,
                row.BEGIN_LON,
                row.END_LON,
            )

            # convert -180-180 -> 0-360
            lon_min, lon_max = (((lon_min + 360) % 360)), (((lon_max + 360) % 360))

            # --- pad to 224, 224 ----
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
            # -----------------------

            # [20, 60]
            lat_mask = (arr.latitude >= lat_min) & (arr.latitude <= lat_max)

            # [230, 300]
            lon_mask = (arr.longitude >= lon_min) & (arr.longitude <= lon_max)

            # crop -> [224, 224]
            arr = arr.where(lat_mask & lon_mask, drop=True)
            cropped_arr = arr.to_array().values.squeeze()

            return cropped_arr
            

        dirs = glob(events_dir + "/*")
        for row in tqdm(df.itertuples(), total=len(df)):

            # get event name
            ymd        = f"{row.BEGIN_YEARMONTH}{row.BEGIN_DAY:02}"
            start_time = row.BEGIN_TIME
            end_time   = row.END_TIME
            scale      = row.TOR_F_SCALE
            event_name = "_".join([ymd, str(start_time), str(end_time), scale])

            files = glob(events_dir + f"/{Path(event_name).parts[-1]}/*.grib2")

            for fp in files:

                try:

                    out_fp = fp.replace(".grib2", "_cropped_224.pkl")

                    # skip existing files
                    if Path(out_fp).is_file(): continue

                    gf = Grib2File(fp)
                    xr = gf.to_xarray()
                    cropped_arr = crop_xarray_to_224_224(row, xr)

                    with open(out_fp, "wb") as f:
                        pickle.dump(cropped_arr, f)

                except:

                    print(f"Error: could not process {fp}")

        # for dir in dirs
            # 1. list all files in sorted order
            # 2.

    def _load_data_from_cache(self):

        import h5py

        # [[event]]
        all_data = {}
        with h5py.File(TornadoMRMSDataset._ALL_DATA_CACHE_FP, "r") as hf:
            for key in tqdm(hf.keys(), desc="🚀 Loading Dataset From Cache"):
                all_data[key] = hf[key][:]

        # NOTE: train/val splits are currently deterministic
        # it may be nice to add optional, random splits in the future
        all_keys = list(sorted(all_data.keys()))

        split_idx = int(0.85 * len(all_keys))
        train_keys = all_keys[:split_idx]
        val_keys = all_keys[split_idx:]

        # assign train/val sets
        self.train_data = [all_data[k] for k in train_keys]
        self.val_data   = [all_data[k] for k in val_keys]

    def _get_dataset_stats(self):

        # calculate minimum and maximum values
        self.max_val = float("-inf")
        data = self.train_data if self.split == "train" else self.val_data
        for idx, arr in enumerate(data):
            try:
                self.max_val = max(self.max_val, float(arr.max()))
            except:
                # remove some erroneous entries
                if self.split == "train":
                    self.train_data.pop(idx)
                else:
                    self.val_data.pop(idx)

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
    dataset = TornadoMRMSDataset(split="train")
    print(len(dataset))
    print(dataset[0])
