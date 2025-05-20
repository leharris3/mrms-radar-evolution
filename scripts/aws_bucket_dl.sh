# cd ..
# aws s3 cp s3://noaa-mrms-pds/CONUS/MergedReflectivityComposite_00.50 \
#     data/MergedReflectivityComposite/2024/ \
#     --exclude "2020*" \
#     --exclude "2021*" \
#     --exclude "2022*" \
#     --exclude "2023*" \
#     --exclude "2025*" \
#     --recursive \
#     --no-sign-request \

#!/usr/bin/env bash
#
# Usage: ./download_mrms.sh path/to/events.csv  [optional_local_root]
# Example: ./download_mrms.sh df_tor.csv data/MergedReflectivityComposite
#
# Requires:  awscli, Python ≥3.6, public-bucket so no AWS creds needed

CSV_PATH="$1"
LOCAL_ROOT="${2:-data/MergedReflectivityComposite}"
S3_PREFIX="s3://noaa-mrms-pds/CONUS/MergedReflectivityComposite_00.50"

# ---------- 1. Build a unique, sorted list of YYYYMMDD strings ----------
mapfile -t DATES < <(python - <<'PY' "$CSV_PATH"
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])

def ym_day_to_date(row):
    ym = f"{int(row.BEGIN_YEARMONTH):06d}"   # e.g. 202402 → "202402"
    year, month = ym[:4], ym[4:]            # "2024", "02"
    day = f"{int(row.BEGIN_DAY):02d}"       # 5 → "05"
    return year + month + day               # "20240205"

dates = {ym_day_to_date(r) for _, r in df[['BEGIN_YEARMONTH','BEGIN_DAY']].iterrows()}
for d in sorted(dates):
    print(d)
PY
)

export AWS_RETRY_MODE=adaptive AWS_MAX_ATTEMPTS=10

for d in "${DATES[@]}"; do
    echo "⏬  Downloading $d ..."
    aws s3 sync --no-sign-request --only-show-errors \
        "${S3_PREFIX}/${d}/"  "${LOCAL_ROOT}/${d}/"
done

echo "✅  All requested dates downloaded to ${LOCAL_ROOT}"