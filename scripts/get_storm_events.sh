#!/usr/bin/env bash
cd ..
for gz in data/StormEvents/csvfiles/*.csv.gz; do
    base=$(basename "${gz%.gz}")          # strip the .gz extension
    gzip -dc "$gz" > "data/StormEvents/unziped/$base"
done