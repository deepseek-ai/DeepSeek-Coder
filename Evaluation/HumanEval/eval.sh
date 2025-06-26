MODEL_NAME_OR_PATH="/scratch/shared_dir/xinyu/deepseek-1.3b"
DATASET_ROOT="data/"
LANGUAGE="rust"
CUDA_VISIBLE_DEVICES=1,2,3 python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}
