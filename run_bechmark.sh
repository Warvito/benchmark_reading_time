#!/bin/bash
# Bechmark reading time using MONAI's different dataset types: Normal, Persistent and Cached.

# Get docker image
docker pull projectmonai/monai

mkdir ./temp
docker run \
  projectmonai/monai:latest \
  "export MONAI_DATA_DIRECTORY=/benchmark/temp/; python /benchmark/monai_benchmark.py" \
  --volume ./:/benchmark \
  | tee output.txt
