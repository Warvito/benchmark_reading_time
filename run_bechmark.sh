#!/bin/bash
# Bechmark reading time using MONAI's different dataset types: Normal, Persistent and Cached.

# Get docker image
docker pull projectmonai/monai

mkdir ./temp
docker run --volume $(pwd):/benchmark --env MONAI_DATA_DIRECTORY="/benchmark/temp/" projectmonai/monai:latest \
  /benchmark/monai_benchmark.py \
  | tee output.txt


docker run \
  projectmonai/monai:latest \
  --volume ./:/benchmark \