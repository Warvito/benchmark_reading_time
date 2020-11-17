# Benchmark reading time

Script to benchmark reading time using project MONAI code.

## Instructions to run on the RUNAI Server

```
runai submit \
  --name monai-benchmark \
  --image projectmonai/monai:0.3.0 \
  --backoffLimit 0 \
  --gpu 0 \
  --large-shm \
  --project wds20 \
  --volume /nfs/home/wds20/projects/benchmark_reading_time:/project \
  --command -- bash run_on_runai_server.sh
```