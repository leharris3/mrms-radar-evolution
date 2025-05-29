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

    def _build_dataset(self): pass

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
            return len(self.train_data)
        else:
            return len(self.val_data)

    def __getitem__(self, idx) -> dict:
        """
        - [6, 224, 224]
        - Normalized -> [0, 1]
        """

        if self.split == "train":
            item: np.ndarray = self.train_data[idx]
        else:
            item: np.ndarray = self.val_data[idx]

        # clip to (0, inf)?
        item = item.clip(0)

        # scale -> [0, 1]
        item = item / self.max_val

        return {
            "item": item,
        }


if __name__ == "__main__":
    dataset = TornadoMRMSDataset(split="train")
    print(len(dataset))
    print(dataset[0])
