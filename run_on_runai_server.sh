#!/bin/bash
printf '%s\n' -------------------- | tee -a $OUTPUT_DIR/logs.txt
echo ENV | tee -a $OUTPUT_DIR/logs.txt
printf '%s\n' -------------------- | tee -a $OUTPUT_DIR/logs.txt
export OUTPUT_DIR=/project/outputs
echo OUTPUT_DIR=$OUTPUT_DIR | tee -a $OUTPUT_DIR/logs.txt

nvidia-smi | tee -a $OUTPUT_DIR/logs.txt


printf '%s\n' --------------------
echo PIP
printf '%s\n' --------------------
pip install -r /project/requirements.txt -q
echo SUCCESS

printf '%s\n' --------------------
echo PYTHON
printf '%s\n' --------------------
python3 /project/monai_benchmark.py | tee -a $OUTPUT_DIR/logs.txt