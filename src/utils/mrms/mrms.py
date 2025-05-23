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

from pathlib import Path
from s3fs import S3FileSystem


class MRMSAWSClient:
    """
    A high-level python API for the AWS MRMS S3 bucket.
    """

    BASE_URL_CONUS = "s3://noaa-mrms-pds/CONUS/"

    def __init__(self, format="NCEP"):

        # create an anonymous fs
        self.s3_file_system = S3FileSystem(anon=True)

    def download(self, path: str, to: str, recursive=False):

        assert self.s3_file_system.exists(path), f"Error! Invalid path: {path}"
        assert Path(to).is_dir(), f"Error! 'To' not a valid dir: {to}"

        # TODO:
        # if  : recursive is true than path msut always be a dir
        # else: path must be a file + file name must be appended to end of "to"

        # try to download files -> "to"
        cmd = ["aws", "s3", "cp", "--no-sign-request", path, str(to)]
        if recursive:
            cmd.append("--recursive")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Download failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        else:
            print(f"✅ Download complete:\n{result.stdout}")

    def get(url: str):
        pass
