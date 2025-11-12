# export NCCL_DEBUG=INFO

MODEL_NAME_OR_PATH="/scratch/shared_dir/xinyu/rust_merged_models/merged_model_full/"
# MODEL_NAME_OR_PATH="/scratch/shared_dir/xinyu/FFT-instruct-Golang-linuxmreitt"
DATASET_ROOT="data/"
LANGUAGE="go"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}
