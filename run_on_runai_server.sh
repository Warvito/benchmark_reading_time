#!/bin/bash
printf '%s\n' --------------------
echo ENV
printf '%s\n' --------------------
export OUTPUT_DIR=/project/outputs
echo OUTPUT_DIR=$OUTPUT_DIR

nvidia-smi | tee -a $OUTPUT_DIR/logs.txt

printf '%s\n' --------------------
echo PYTHON
printf '%s\n' --------------------
python3 /project/monai_benchmark.py | tee -a $OUTPUT_DIR/logs.txt