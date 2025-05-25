"""
# Sources
---
- https://github.com/NOAA-National-Severe-Storms-Laboratory/mrms-support
- https://training.weather.gov/wdtd/courses/MRMS/lessons/overview-v12.2/presentation_html5.html
- https://github.com/HumphreysCarter/mrms-api

# Dataset Structure
---
- DOMAIN
    - PRODUCT
        - YYYYMMDD
            - PRODUCT_YYYYMMDD-ZZZZZZ.grib2.gz
"""

import xarray
import subprocess

from enum import Enum
from typing import List
from pathlib import Path
from s3fs import S3FileSystem


class MRMSURLs:

    BASE_URL = "s3://noaa-mrms-pds/"
    BASE_URL_CONUS = "s3://noaa-mrms-pds/CONUS/"


class MRMSProducts:
    """
    An enumeration of all available MRMS CONUS products at a given moment.
    """

    def __init__(self):
        self.products = MRMSProducts._fetch_products()

    @staticmethod
    def _fetch_products() -> List[str]:
        s3_file_system = S3FileSystem(anon=True)
        results = s3_file_system.ls(MRMSURLs.BASE_URL_CONUS)
        products = []
        for res in results:
            products.append(res.split('/')[-1])
        return products


class MRMSAWSS3Client:
    """
    A high-level python API for the public MRMS AWS S3 bucket.
    """

    def __init__(self, format="NCEP"):

        # create an anonymous fs
        self.s3_file_system = S3FileSystem(anon=True)

    def ls(self, path: str) -> List[str]:
        return self.s3_file_system.ls(path)

    def download(self, path: str, to: str, recursive=False) -> List[str]:
        """
        Returns
        ---
        - A list of `str` paths corresponding to successfully downloaded files.
        """

        assert self.s3_file_system.exists(path), f"Error! Invalid path: {path}"
        # assert Path(to).is_dir(), f"Error! 'To' not a valid dir: {to}"

        # if  : recursive is true than path msut always be a dir
        # else: path must be a file + file name must be appended to end of "to"
        remote_files = [path]
        if recursive == True:
            assert path.endswith("/"), (
                "When recursive=True the S3 path must end with '/' so it is "
                "interpreted as a prefix, not a single object."
            )
            remote_entries = self.s3_file_system.ls(path, detail=True)
            remote_files = [e["Key"] for e in remote_entries if e["type"] == "file"]
        else:
            assert not path.endswith("/"), (
                "When recursive=False the S3 path must point to a single object, "
                "not a directory prefix."
            )

        # map remote keys to download dst paths
        dst_root = Path(to).expanduser().resolve()
        local_paths: List[str] = []
        if recursive:
            prefix_len = len(path)
            for key in remote_files:
                rel_key = key[prefix_len:]
                local_paths.append(str(dst_root / rel_key))
        else:
            local_paths.append(str(dst_root / Path(path).name))

        # try to download files -> "to"
        cmd = ["aws", "s3", "cp", path, str(to), "--no-sign-request"]
        if recursive:
            cmd.append("--recursive")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Download failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        return local_paths

    def submit_bulk_download(self, paths: List[str], tos: List[str]): ...


if __name__ == "__main__":
    prod = MRMSProducts()
    client = MRMSAWSS3Client()
    res = client.ls(MRMSURLs.BASE_URL_CONUS)
    breakpoint()
