MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1b-base/"
DATASET_ROOT="data/"
LANGUAGE="python"
CUDA_VISIBLE_DEVICES=1,2,3 python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}