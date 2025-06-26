LANG="rust"
OUTPUT_DIR="output"
MODEL="/scratch/shared_dir/xinyu/deepseek-6.7b-instruct"
MODEL_NAME=$(basename "$MODEL")

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0,1 python eval_instruct.py \
    --model "$MODEL" \
    --output_path "$OUTPUT_DIR/${LANG}.${MODEL_NAME}.jsonl" \
    --language $LANG \
    --temp_dir $OUTPUT_DIR
