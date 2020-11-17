#!/bin/bash
# Bechmark reading time using MONAI's different dataset types: Normal, Persistent and Cached.

# Get docker image
docker pull projectmonai/monai

chmod +x $(pwd)/monai_benchmark.py

mkdir ./temp
docker run --volume $(pwd):/opt/monai/benchmark --env MONAI_DATA_DIRECTORY="/opt/monai/benchmark/temp/" projectmonai/monai:latest \
  /opt/monai/benchmark/monai_benchmark.py \
  | tee output.txt


