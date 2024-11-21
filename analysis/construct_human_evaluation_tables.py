
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import re
import logging
import math
from tqdm import tqdm
import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from bench_generation.utils.tools import extract_json_from_string, load_json_file, save_json_file
from bench_generation.utils.openai_models import OpenAIClientWrapper
import argparse
import os
import numpy as np
import glob
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Quality Control for Generated Questions.")
    parser.add_argument('--file_dir', type=str, default='./outputs/data/March-2024-to-September-2024/',
                        help='Path to the file containing generated questions.')
    return parser.parse_args()

def reorganization(dir_):

    sum_file_name = os.path.join(dir_, 'documents.json')
    documents_data = load_json_file(sum_file_name)
    document_indices = list(range(len(documents_data)))
    random.seed(2024)
    random.shuffle(document_indices)

    all_files = glob.glob(os.path.join(dir_, '*'))
    selected_files = [f for f in all_files if 'full_haystack_question' in os.path.basename(f)]
    all_files = glob.glob(os.path.join("./results/March-2024-to-September-2024/", '*'))
    results_files = [f for f in all_files if '128000' in os.path.basename(f)]
    # print ("selected_files:", selected_files)
    name_dict = {
            'sssd': 'SingleStepSingleDocumentTask',
            '2ssd': 'TwoStepSingleDocumentTask',
            '3ssd': 'ThreeStepSingleDocumentTask',
            'ss2d': 'SingleStepTwoDocumentTask',
            '2s2d': 'TwoStepTwoDocumentTask',
            '3s2d': 'ThreeStepTwoDocumentTask',
            '3s3d': 'ThreeStepThreeDocumentTask'
            }
    
    data_need ={
        'sssd': 25,
        '2ssd': 25,
        '3ssd': 25,
        'ss2d': 25,
        '2s2d': 25,
        '3s2d': 25,
        '3s3d': 25,
    }
    count = 0
    for task_name in data_need:

        for selected_file in selected_files:
            if task_name in selected_file:
                data_stat_file = selected_file
        for results_file in results_files:
            if task_name in results_file:
                oresult_file = results_file
        
        print (data_stat_file)
        print (oresult_file)

        # data_stat_rows = load_json_file(selected_file)
        ### 
        # each of data_stat_rows include dict_keys(['Topic', 'Subtopic', 'Query', 'Document_ID', 'Documents', 'task_type', 'Task', 'Irrelevant_Documents_Indexs'])
        # in Task column, which include dict_keys(['relevant_quantity_cells_from_two_documents', 'question', 'solution', 'steps', 'answer', 'refined_flag', 'consistency'])
        ###

        data_result_rows = load_json_file(oresult_file)
        ###
        # each of data_result_rows include dict_keys(['index', 'relevant_documents', 'question', 'expected_answer', 'pred_answer', 'correct', 'llm_annotate_reasoning', 'llm_judge', 'pred_reasoning', 'judge_reasoning'])
        ### 
        # kept_data_stat_rows = []
        # for data_stat_row in data_stat_rows:
        #     if data_stat_row["Task"]["consistency"] == 1:
        #         kept_data_stat_rows.append(data_stat_row)
        print (len(data_result_rows))

        # assert 1==0
        
        
        combined_rows = []
        # Merge data for CSV generation
        need_c = data_need[task_name]
        # if task_name == "3s3d":
        #     need_c = 100
        count = 0
        non_c = 0
        for result_row in data_result_rows[:100]:
            if task_name in ["ss2d", "2s2d", "3s2d"]:
                relevant_quantity_cells = result_row['Task']["relevant_quantity_cells_from_two_documents"]
            elif task_name in ["sssd", "2ssd", "3ssd"]:
                relevant_quantity_cells = result_row['Task']["relevant_quantity_cells"]
            elif task_name in ["3s3d"]:
                relevant_quantity_cells = result_row['Task']["relevant_quantity_cells_from_three_documents"]
            if task_name in ["3s2d", "3ssd", "3s3d"]:
                if "Step 3" not in  result_row['Task']['solution']:
                    continue
            if task_name in ["2s2d", "2ssd"]:
                if "Step 2" not in  result_row['Task']['solution']:
                    continue
                if "Step 3" in  result_row['Task']['solution']:
                        continue
                    # continue
            combined_row = {
                "document": result_row['Documents'],
                "relevant_quantity_cells": relevant_quantity_cells,
                "generated_question": result_row['Task']['question'],
                "generated_solution": result_row['Task']['solution'],
                "reasoning_steps_from_python_solution": result_row['Task']['steps'],
                "answer": result_row['expected_answer'],
                "predicted_solution_by_GPT-4o": result_row['pred_reasoning'],
                "predicted_answer_by_GPT-4o": result_row['pred_answer'],
                # "GPT-4o_evaluation_for_prediced_answer": result_row['llm_judge'],
                "llm_judge": result_row['correct'],
                "relevance_of_question_annotation (1:weak|2:medium|3:strong)": "",
                "is_question_natural_annotation (1:weak|2:medium|3:strong)": "",
                "consistency_between_question_and_solution_annotation (Yes|No)": "",
                "how_many_steps_annotation (1|2|3)": "",
                "whether_question_can_have_multiple_answers (Yes|No)": "",
                "human_evaluation_for_prediced_answer (Yes|No)": ""
            }
            combined_rows.append(combined_row)
            count += 1
            if count>=25:
                break
        print (task_name, count)
    
        # Generate CSV
        csv_file_path = os.path.join("./analysis/annotation_files/", "human_evaluation_annotations_"+task_name+".csv")
        xlsx_file_path = os.path.join("./analysis/annotation_files/", "human_evaluation_annotations_"+task_name+".xlsx")
        df = pd.DataFrame(combined_rows)
        # df.to_csv(csv_file_path, index=False)
        df.to_excel(xlsx_file_path, index=False, engine='openpyxl')


        logger.info(f"XLSX file generated at: {xlsx_file_path}\n")


if __name__ == "__main__":
    args = parse_args()
    
    # Perform quality control on the loaded data
    reorg_df = reorganization(args.file_dir)