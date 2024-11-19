#!/bin/bash

# sh scripts/bench_generation.sh March-2024-to-September-2024 10 2 5 sssd
# sh scripts/bench_generation.sh March-2024-to-September-2024 10 2 5 2ssd
# sh scripts/bench_generation.sh March-2024-to-September-2024 10 2 5 3ssd
# sh scripts/bench_generation.sh March-2024-to-September-2024 10 2 5 ss2d
# sh scripts/bench_generation.sh March-2024-to-September-2024 10 2 5 2s2d
# sh scripts/bench_generation.sh March-2024-to-September-2024 10 2 5 3s2d
# sh scripts/bench_generation.sh March-2024-to-September-2024 10 2 5 3s3d
# Assign input arguments to variables
TIME=$1
NUM_TOPICS=$2
NUM_SUBTOPICS=$3
NUM_QUERIES=$4
TASKX=$5

# ### topic generation
# python -m bench_generation.document_collection.topic_generation \
#     --model_name gpt-4o \
#     --num_topics $NUM_TOPICS \
#     --llm_batch_size 10 \
#     --save_path './outputs/data/' \
#     # --generate_topic_flag 

# ### subtopics and thier queries generation
# python -m bench_generation.document_collection.query_generation \
#     --model_name gpt-4o \
#     --num_subtopics $NUM_SUBTOPICS \
#     --num_queries $NUM_QUERIES \
#     --time_period "$TIME"\
#     --llm_batch_size 5 \
#     --save_path ./outputs/data/ \
#     --main_topics_file ./outputs/data/main_topics.json \
#     # --generate_subtopics_flag

# ### document search by tavily
# python -m bench_generation.document_collection.document_search \
#     --model_name gpt-4o \
#     --min_numbers 10 \
#     --min_sentences 10 \
#     --min_words 400 \
#     --min_entities 10 \
#     --save_path ./outputs/data/ \
#     --time_period "$TIME" \
#     # --generate_documents_flag 

# # # Define paths based on time period
# FILTERED_DOCUMENTS_FILE="./outputs/data/${TIME}/filtered_documents.json"
# OUTPUT1_FILE="./outputs/data/${TIME}/summarized_documents.json"
# OUTPUT2_FILE="./outputs/data/${TIME}/documents.json"

# python -m bench_generation.document_collection.document_summarization \
#     --model_name gpt-4o \
#     --llm_batch_size 10 \
#     --save_path ./outputs/data/ \
#     --filtered_documents_file "$FILTERED_DOCUMENTS_FILE" \
#     --output_file "$OUTPUT1_FILE" \
#     --document_file "$OUTPUT2_FILE" \
#     --generate_summary_flag


# Define paths based on time period
SUMMARIZED_DOCUMENTS_FILE="./outputs/data/${TIME}/summarized_documents.json"
DOCUMENTS_FILE="./outputs/data/${TIME}/documents.json"


# List of TASK values
TASKS=("${TASKX}") #"sssd" "2ssd" "3ssd"  "ss2d" "2s2d" "3s2d" "3s3d" 

# Loop through each TASK and run the Python command
for TASK in "${TASKS[@]}"
do
  # Set the output file based on the TASK
  OUTPUT_FILE="./outputs/data/${TIME}/generated_questions_${TASK}.json"
  echo "question_generation_$TASK"
  # Execute the question generation script for each TASK
  python -m bench_generation.question_generation.question_generation_${TASK} \
      --model_name gpt-4o \
      --llm_batch_size 10 \
      --summarized_documents_file "$SUMMARIZED_DOCUMENTS_FILE" \
      --output_file "$OUTPUT_FILE" \
      --generate_questions_flag


  GENERATED_QUESTIONS_FILE="./outputs/data/${TIME}/generated_questions_${TASK}.json"
  HIGH_QUALITY_QUESTIONS_FILE="./outputs/data/${TIME}/high_quality_questions_${TASK}.json"

  # Execute the quality control script
  python -m bench_generation.quality_control.quality_control \
      --input_file "$GENERATED_QUESTIONS_FILE" \
      --output_file "$HIGH_QUALITY_QUESTIONS_FILE" \
      --model_name gpt-4o \
      --task_name "$TASK" \
      --quality_control_flag

done

FILE_NAME="./outputs/data/${TIME}/"

python -m bench_generation.haystack_construction.haystack --file_dir "$FILE_NAME"