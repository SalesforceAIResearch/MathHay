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
    selected_files = [f for f in all_files if 'high_quality_questions' in os.path.basename(f)]

    name_dict = {'sssd': 'SingleStepSingleDocumentTask',
            '2ssd': 'TwoStepSingleDocumentTask',
            '3ssd': 'ThreeStepSingleDocumentTask',
            'ss2d': 'SingleStepTwoDocumentTask',
            '2s2d': 'TwoStepTwoDocumentTask',
            '3s2d': 'ThreeStepTwoDocumentTask',
            '3s3d': 'ThreeStepThreeDocumentTask'}
    
    final_data ={
        'sssd': [],
        '2ssd': [],
        '3ssd': [],
        'ss2d': [],
        '2s2d': [],
        '3s2d': [],
        '3s3d': []
    }
    count = 0
    for selected_file in selected_files:
        task_name = ''
        for k, v in name_dict.items():
            if k in selected_file:
                task_name = k
        data_list = load_json_file(selected_file)
        print (selected_file, task_name, len(data_list))

        for index, elem in enumerate(data_list):
        # for index, row in tqdm(df.iterrows(), total=len(df)):
            tasks = elem["tasks"]
            if type(elem["doc_id"]) == str:
                doc_ids = [elem["doc_id"]]
                documents = [elem["document"]]
            else:
                doc_ids = elem["doc_id"]
                documents = elem["document"]

            for task in tasks:
                irrelevant_documents_indexs = []
                for doc_id in document_indices:
                    if "Doc_"+str(doc_id) in doc_ids:
                        continue
                    irrelevant_documents_indexs.append(doc_id)

                data_row = {
                    "Topic": elem["Topic"],
                    "Subtopic": elem["Subtopic"],
                    "Query": elem["decomposable_query"],
                    "Document_ID": doc_ids,
                    "Documents": documents,
                    "task_type": task_name,
                    "Task": task,
                    "Irrelevant_Documents_Indexs": irrelevant_documents_indexs,
                }
                    
                final_data[task_name].append(data_row)

    topics = load_json_file(os.path.join("./outputs/data/", "main_topics.json"))

    for name, data_rows in final_data.items():
        # if name!="sssd":
        #     continue
        topic_count_dict = {}
        for topic in topics:
            topic_count_dict[topic] = 0
        print (f"{name} : {len(data_rows)}")
        if data_rows == []:
            continue

        to_verify_data_rows = []
        unverified_data_rows = []
        data_rows_full  = []
        for row in data_rows:
            rtopic = row['Topic']
            if topic_count_dict[rtopic] >=2 and topic_count_dict[rtopic] <12:
                unverified_data_rows.append(row)
                data_rows_full.append(row)
            elif topic_count_dict[rtopic] <2:
                to_verify_data_rows.append(row)
                data_rows_full.append(row)
            else:
                pass
            topic_count_dict[rtopic]+=1
        print (f"topic_count_dict: {topic_count_dict}")

        
        print (f"{name} data_rows_full: {len(data_rows_full)}, unverified_data_rows: {len(unverified_data_rows)}, to_verify_data_rows: {len(to_verify_data_rows)},")
        sum_file_name = os.path.join(dir_, f'full_haystack_question_{name}.json')
        save_json_file(sum_file_name, data_rows_full)
        logging.info(f"{sum_file_name} saving done.")
        
        to_verify_sum_file_name = os.path.join(dir_, f'to_verify_haystack_question_{name}.json')
        print (f"{to_verify_sum_file_name} : {len(to_verify_data_rows)}")
        save_json_file(to_verify_sum_file_name, to_verify_data_rows)
        logging.info(f"{to_verify_sum_file_name} saving done.")

        unerified_sum_file_name = os.path.join(dir_, f'unverified_haystack_question_{name}.json')
        save_json_file(unerified_sum_file_name, unverified_data_rows)
        logging.info(f"{unerified_sum_file_name} saving done.")


if __name__ == "__main__":
    args = parse_args()
    
    # Perform quality control on the loaded data
    reorg_df = reorganization(args.file_dir)

    logger.info(f"Reorganized questions saved to {args.file_dir}")