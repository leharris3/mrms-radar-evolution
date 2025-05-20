import pandas as pd
from torch.utils.data.dataset import Dataset
from pathlib import Path
from typing import Tuple

BASEDIR    = "data/MergedReflectivityComposite"
EVENTS_DIR = "data/2024-tornado-MergedReflectivityComposite-0.5km/events"
METADATA   = "data/2024-tornado-MergedReflectivityComposite-0.5km/metadata/tornados.csv"


class TornadoReflectivityDataset(Dataset):
    """
    Dataset of 2024 tornado events with MRMS 0.5km reflectivity data.
    """

    def __init__(self, split: str = "train"):
        super().__init__()
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.metadata = pd.read_csv(METADATA)

    def _get_tornado_data_pair(row) -> Tuple[str, str]:

        yearmonth = str(row[1].BEGIN_YEARMONTH)
        day       = int(row[1].BEGIN_DAY)
        day       = f"{day:02d}"

        subdir_name = yearmonth + day
        
        # reflectivity_dir = Path(BASEDIR) / subdir_name
        start_time = row[1].BEGIN_TIME
        end_time   = row[1].END_TIME
        scale      = row[1].TOR_F_SCALE

        event_name = '_'.join([subdir_name, str(start_time), str(end_time), scale])

        # path to events dir
        events_dir = Path(EVENTS_DIR) / event_name

        data_paths = sorted([str(f) for f in events_dir.glob('*.grib2')])
        return data_paths[0], data_paths[1]

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    dataset = TornadoReflectivityDataset(split="train")
    print(len(dataset))
    print(dataset[0])