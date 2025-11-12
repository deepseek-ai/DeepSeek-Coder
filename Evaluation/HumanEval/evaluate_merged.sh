#!/bin/bash

# --- Configuration ---
BASE_MODEL_DIR="/scratch/shared_dir/xinyu/rust_merged_models"

OUTPUT_LOG="/scratch/shared_dir/xinyu/merged_models_evaluation_results.csv"

LANGUAGES=("rust" "go")

DATASET_ROOT="data/"
CONFIG_FILE="test_config.yaml"

GPU_DEVICES="0,1"

NUM_RUNS_PER_EVAL=1


echo "🔍 Finding all model directories under ${BASE_MODEL_DIR}..."
MODEL_DIRS=$(find "$BASE_MODEL_DIR" -type f -name "config.json" -exec dirname {} \; | sort -u)

if [ -z "$MODEL_DIRS" ]; then
    echo "❌ Error: No models found in ${BASE_MODEL_DIR}. Please check the path."
    exit 1
fi

NUM_MODELS=$(echo "$MODEL_DIRS" | wc -l)
NUM_LANGUAGES=${#LANGUAGES[@]}
TOTAL_RUNS=$((NUM_MODELS * NUM_LANGUAGES * NUM_RUNS_PER_EVAL))

echo "✅ Found ${NUM_MODELS} models. Each evaluation will run ${NUM_RUNS_PER_EVAL} time(s)."
echo "   Total evaluations to run: ${TOTAL_RUNS}."
echo ""

echo "model_name,category,language,run_number,score" > "$OUTPUT_LOG"
echo "📝 Results will be saved to ${OUTPUT_LOG}"
echo ""

completed_runs=0
start_time_total=$(date +%s)

for model_path in $MODEL_DIRS; do
    for lang in "${LANGUAGES[@]}"; do
        for (( run_i=1; run_i<=${NUM_RUNS_PER_EVAL}; run_i++ )); do
            
            completed_runs=$((completed_runs + 1))
            model_name=$(basename "$model_path")

            category="unknown"
            case "$model_path" in
              */merged_model_full)           category="full" ;;
              */merged_model_one_layer/*)    category="one_layer" ;;
              */merged_models_window_2/*)  category="window_2" ;;
              */merged_models_window_3/*)  category="window_3" ;;
            esac

            echo "========================================================================"
            echo "🚀 Starting evaluation ${completed_runs} of ${TOTAL_RUNS}"
            echo "   - Model:    ${model_name}"
            echo "   - Category: ${category}"
            echo "   - Language: ${lang}"
            echo "   - Run:      ${run_i} of ${NUM_RUNS_PER_EVAL}"
            echo "========================================================================"


            set -o pipefail
            output=$(CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m accelerate.commands.launch --config_file ${CONFIG_FILE} eval_pal.py --logdir "${model_path}" --language "${lang}" --dataroot "${DATASET_ROOT}" 2>&1)
            exit_code=$?
            set +o pipefail


            score=""
            if [ $exit_code -eq 0 ]; then
                score=$(echo "$output" | grep "Score is" | awk -F'is ' '{print $2}')
                if [ -z "$score" ]; then
                    score="PARSING_ERROR" 
                    echo "⚠️ Warning: Evaluation script finished but score could not be parsed."
                fi
            else
                score="EXECUTION_ERROR"
                echo "❌ ERROR: Evaluation script failed with exit code ${exit_code}."
            fi


            echo "${model_name},${category},${lang},${run_i},${score}" >> "$OUTPUT_LOG"
            echo "   - Score:    ${score}"
            echo "   - Result logged to ${OUTPUT_LOG}"


            current_time=$(date +%s)
            elapsed_time=$((current_time - start_time_total))
            
            if [ $completed_runs -gt 0 ]; then
                avg_time_per_run=$((elapsed_time / completed_runs))
                remaining_runs=$((TOTAL_RUNS - completed_runs))
                estimated_seconds_remaining=$((remaining_runs * avg_time_per_run))
                

                est_h=$((estimated_seconds_remaining / 3600))
                est_m=$(( (estimated_seconds_remaining % 3600) / 60 ))
                est_s=$((estimated_seconds_remaining % 60))

                echo "   - Time elapsed: ${elapsed_time}s"
                if [ $remaining_runs -gt 0 ]; then
                     echo "   - Estimated time remaining: ${est_h}h ${est_m}m ${est_s}s ⏳"
                fi
            fi
            echo ""
        
        done
    done
done


total_duration=$(( $(date +%s) - start_time_total ))
total_h=$((total_duration / 3600))
total_m=$(( (total_duration % 3600) / 60 ))
total_s=$((total_duration % 60))

echo "🎉🎉🎉 All ${TOTAL_RUNS} evaluations finished! 🎉🎉🎉"
echo "Total execution time: ${total_h}h ${total_m}m ${total_s}s."
echo ""
echo "=============== FINAL RESULTS SUMMARY ==============="

if command -v column &> /dev/null; then
    column -t -s, "$OUTPUT_LOG"
else
    cat "$OUTPUT_LOG"
fi
echo "==================================================="
echo "Full results are saved in ${OUTPUT_LOG}"