MODEL_NAME_OR_PATH="/scratch/shared_dir/xinyu/FFT-instruct-Rust-ammar-200k"
DATASET_ROOT="data/"
LANGUAGE="python"
CUDA_VISIBLE_DEVICES=0,1,2 python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}
