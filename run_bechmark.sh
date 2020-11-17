#!/bin/bash
# Bechmark reading time using MONAI's different dataset types: Normal, Persistent and Cached.

# Get docker image
docker pull projectmonai/monai:0.3.0

chmod +x $(pwd)/monai_benchmark.py

nvidia-docker run --volume $(pwd):/project --env OUTPUT_DIR="/project/outputs" projectmonai/monai:0.3.0 \
  /project/monai_benchmark.py