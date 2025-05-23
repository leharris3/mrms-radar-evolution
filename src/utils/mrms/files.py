import gzip
import shutil
from pathlib import Path


class Grib2File:

    def __init__(self, path: str):
        self.path = Path(path)
        assert self.path.suffix == ".grib2"


class ZippedGrib2File:

    def __init__(self, path: str):
        self.path = Path(path)
        assert self.path.suffix == ".gz"

    def unzip(self, to_dir: str) -> str:
        to_dir = Path(to_dir)
        assert to_dir.exists(), f"Error! Bad path: {str(to_dir)}"
        assert to_dir.is_dir(), f"Error! Invalid dir path: {str(to_dir)}"
        # TODO: make this a little more robust
        dst_fp = to_dir / Path(self.path.name.replace(".gz", ""))
        # unzip and write to out
        with gzip.open(str(self.path), "rb") as rp:
            with open(str(dst_fp), "wb") as wp:
                shutil.copyfileobj(rp, wp)
        return str(to_dir)
