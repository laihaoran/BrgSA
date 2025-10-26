CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python zero_shot_3d.py

python ./bootstrap_evaluation/bootstrap_performance.py --mainpath ./bootstrap_evaluation/CT-RATE

python ./bootstrap_evaluation/get_metric.py --mainpath ./bootstrap_evaluation/CT-RATE