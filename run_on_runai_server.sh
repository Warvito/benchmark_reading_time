#!/bin/bash
printf '%s\n' --------------------
echo ENV
printf '%s\n' --------------------

export OUTPUT_DIR=file:/project/outputs
echo OUTPUT_DIR=$OUTPUT_DIR

printf '%s\n' --------------------
echo PYTHON
printf '%s\n' --------------------
python3 /project/monai_benchmark.py | tee $OUTPUT_DIR/logs.txt