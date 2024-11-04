#!/bin/bash

# Usage: ./run_evaluation.sh [TIME] [TASK] [TASKNAME]
# Example: ./run_evaluation.sh "September-2024" "quality_control" "task_name_example"

# Assign input arguments to variables
TIME=$1 # "March-2024-to-September-2024"
TASK=$2 #sssd

# Set up model parameters
MODEL_NAME=$3 # "gpt-4o"
LLM_BATCH_SIZE=5
HAYSTACK_LEN=$4 # "32000"
PLACEMENT=$5 #"middle"
MODE=$6 # "verified"
INPUT_DIR="./outputs/data/${TIME}/"
OUTPUT_DIR="./results/${TIME}/"
INPUT_FILE="${MODE}_haystack_question_${TASK}.json"

# Run the evaluation code
python -m evaluation.evaluation \
    --model_name "$MODEL_NAME" \
    --llm_batch_size "$LLM_BATCH_SIZE" \
    --task "$TASK" \
    --haystack_len "$HAYSTACK_LEN" \
    --placement "$PLACEMENT" \
    --mode "$MODE" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --input_file "$INPUT_FILE"

# sh scripts/evaluation.sh March-2024-to-September-2024 sssd gpt-4o 32000 middle full
# sh scripts/evaluation.sh March-2024-to-September-2024 2ssd gpt-4o 32000 middle full
# sh scripts/evaluation.sh March-2024-to-September-2024 3ssd gpt-4o 32000 middle full
# sh scripts/evaluation.sh March-2024-to-September-2024 ss2d gpt-4o 32000 first-middle full
# sh scripts/evaluation.sh March-2024-to-September-2024 2s2d gpt-4o 32000 first-middle full
# sh scripts/evaluation.sh March-2024-to-September-2024 3s2d gpt-4o 32000 first-middle full
# sh scripts/evaluation.sh March-2024-to-September-2024 3s3d gpt-4o 32000 first-middle-last full

# sh scripts/evaluation.sh March-2024-to-September-2024 sssd gpt-4o 32000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 sssd gpt-4o 64000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 sssd gpt-4o 128000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 2ssd gpt-4o 32000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 2ssd gpt-4o 64000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 2ssd gpt-4o 128000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3ssd gpt-4o 32000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3ssd gpt-4o 64000 middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3ssd gpt-4o 128000 middle verified

# sh scripts/evaluation.sh March-2024-to-September-2024 ss2d gpt-4o 32000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 ss2d gpt-4o 64000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 ss2d gpt-4o 128000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 2s2d gpt-4o 32000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 2s2d gpt-4o 64000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 2s2d gpt-4o 128000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3s2d gpt-4o 32000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3s2d gpt-4o 64000 first-middle verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3s2d gpt-4o 128000 first-middle verified

# sh scripts/evaluation.sh March-2024-to-September-2024 3s3d gpt-4o 32000 first-first-first verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3s3d gpt-4o 64000 first-first-first verified
# sh scripts/evaluation.sh March-2024-to-September-2024 3s3d gpt-4o 128000 first-first-first verified