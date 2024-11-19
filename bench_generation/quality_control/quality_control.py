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
from bench_generation.utils.openai_models import OpenAIClientWrapper
from bench_generation.utils.tools import extract_json_from_string, load_or_generate, load_json_file, save_json_file
import argparse
import os
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Quality Control for Generated Questions.")
    parser.add_argument('--input_file', type=str, default='./outputs/data/generated_questions_3s2d.csv',
                        help='Path to the file containing generated questions.')
    parser.add_argument('--output_file', type=str, default='./outputs/data/high_quality_questions_3s2d.csv',
                        help='Path to save the high-quality questions after quality control.')
    parser.add_argument('--model_name', type=str, default='gpt-4o', 
                        help='Name of the model to use for quality control verification.')
    parser.add_argument('--task_name', type=str, default='SingleStepSingleDocumentTask', 
                        help='Name of the model to use for quality control verification.')
    parser.add_argument('--quality_control_flag', action='store_true', 
                        help='If set, perform quality control on the generated questions.')
    return parser.parse_args()


# Define the Pydantic model for reasoning and answer verification
class ReasoningVerification(BaseModel):
    reasoning: str = Field(
        description="solution process."
    )
    python_solution: str = Field(
        description="A Python function that solves the generated question using one or several arithmetic operations. The function must be executable, with clearly named variables reflecting the extracted information and a result assigned to a variable named `answer`. The solution demonstrates the reasoning process leading to the final answer."
    )
    answer: float = Field(
        description="The final numerical answer to the question, deduced through reasoning."
    )

def execute_python_solution(solution_code: str) -> float:
    """
    Execute the Python solution code and return the computed answer.
    
    :param solution_code: The Python code as a string.
    :return: The computed answer as a float.
    """
    try:
        # Execute the provided solution code in a controlled environment
        local_vars = {}
        exec(solution_code, {}, local_vars)
        return local_vars.get('answer', None)
    except Exception as e:
        logger.error(f"Error executing solution code: {e}")
        return None

def execute_function_from_string(func_string):
    # Prepare a custom namespace for exec
    local_namespace = {}
    
    # Execute the function string to define the function in the custom namespace
    exec(func_string, globals(), local_namespace)

    # Extract the function name using regex
    match = re.search(r'def (\w+)\s*\(', func_string)
    if match:
        function_name = match.group(1)
        # Dynamically call the function using the custom namespace
        if function_name in local_namespace:
            try:
                return local_namespace[function_name]()
            except:
                return -float("inf")
                # raise KeyError(f"Function '{function_name}' is not defined in the current scope.")
        else:
            raise KeyError(f"Function '{function_name}' is not defined in the current scope.")
    else:
        raise ValueError("No function definition found in the provided string.")
    return -float("inf")


def get_quality_control_prompt_template() -> str:
    """
    Provides a custom prompt template for the quality control verification process.
    
    :return: A string template for the prompt.
    """
    return """
You are tasked with solving a mathematical reasoning question using information from the given document. 
Use the given revelvant documents and quantity cells to solve the given question. Ensure your solution involves a single or mutiple computational steps based on the relevant data extracted. Focus on arithmetic operations as required by the question.

Instructions:
1. **Provide a Python Solution**: Write a Python function that solves the question using basic arithmetic or logical steps. The function should:
   - Be executable by a Python interpreter.
   - Avoid using arguments in the function definition; instead, variables must be named and assigned appropriately based on the given documents and quantity cells.
   - Assign the computed result to a variable named `answer` and ensure the function returns the `answer` variable.
2. **Determine the Final Answer**: The final answer should be presented as an Arabic numeral.

Example Python function format:
```python
def solve():
    # Extracted and relevant quantity cells from Document 1
    variable_1 = numerical_value_1  # Object 1 attribute from Document 1
    # Extracted and relevant quantity cells from Document 2
    variable_2 = numerical_value_2  # Object 2 attribute from Document 2
    # First step of computation
    intermediate_result = variable_1 - variable_2  # Replace with actual first step logic
    # Second step of computation
    answer = intermediate_result * some_other_value  # Replace with actual second step logic
    return answer

Relevant Documents:
{documents}

Relevant Quantity Cells:
{quantity_cells}

Question:
{question}

Output:
- {format_instructions}
"""

def verify_with_llm(llm, question, relevant_quantity_cells, documents) -> float:
    """
    Verify the answer by asking the LLM to solve the problem based on the Pydantic model task.
    
    :param task: The ReasoningVerification object containing the question and relevant quantity cells.
    :return: The answer generated by the LLM.
    """
    # Use the PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=ReasoningVerification)
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template=get_quality_control_prompt_template(),
        input_variables=["question", "quantity_cells", "documents", "format_instructions"]
    )
    
    # Construct the prompt using the provided question and relevant quantity cells
    prompt = prompt_template.format(
        question=question,
        quantity_cells=json.dumps(relevant_quantity_cells, indent=2),
        documents=documents,
        format_instructions=parser.get_format_instructions()
    )
    message_list = [{"role": "user", "content": prompt}]
    response = llm.call_llm_api(message_list, temperature=0, max_tokens=1024)

    try:
        response_json = extract_json_from_string(response)
        verified_task = ReasoningVerification(**response_json)
    except:
        return -float("inf")

    python_solution = verified_task.python_solution
    answer = verified_task.answer
    final_answer = execute_function_from_string(python_solution)
    try:
        final_answer = execute_function_from_string(python_solution)
        logger.info("execute_function_from_string")
        return final_answer
    except:
        logger.info("answer from llm")
        return answer

def quality_control(llm, data_list: pd.DataFrame, name="SingleStepSingleDocumentTask", quality_control_flag=True, save_path="high_quality_sssd_gen.json") -> pd.DataFrame:

    """
    Perform quality control on the generated data examples from a DataFrame.
    
    :param df: DataFrame containing the generated tasks in a JSON format under the 'SingleStepSingleDocumentTask' column.
    :return: A filtered DataFrame with high-quality data examples.
    """
    if not quality_control_flag:
        # Load high-quality DataFrame from a saved CSV file
        high_quality_rows = load_json_file(save_path)
        logger.info(f"Loaded high-quality examples from {save_path}")
        return high_quality_rows


    high_quality_rows = []

    filtered_count = 0
    original_count = 0

    for index, elem in enumerate(data_list):
        tasks = elem["tasks"]

        tasks_filterd = []

        for task in tasks:
            # Execute the provided solution and get the answer
            solution_code = task.get("solution", "")
            generated_answer = task.get("answer", None)
            try:
                computed_answer = execute_function_from_string(solution_code) #execute_python_solution(solution_code)
            except:
                logging.info("execution error")
                continue
                
            # print ("solution_code\n", solution_code)
            # print ("computed_answer:", computed_answer)

            # Verify with LLM
            if name in ["SingleStepTwoDocumentTask", "TwoStepTwoDocumentTask", "ThreeStepTwoDocumentTask"]:
                documents = elem["document"]
                llm_answer = verify_with_llm(llm, task.get("question", ""), task.get("relevant_quantity_cells_from_two_documents", {}), documents)
            elif name in ["ThreeStepThreeDocumentTask"]:
                documents = elem["document"]
                llm_answer = verify_with_llm(llm, task.get("question", ""), task.get("relevant_quantity_cells_from_three_documents", {}), documents)
            else:
                documents = elem["document"]
                if type(documents) == str:
                    documents = [documents]
                llm_answer = verify_with_llm(llm, task.get("question", ""), task.get("relevant_quantity_cells", {}), documents)

            try:
                if computed_answer is None or llm_answer is None or math.isclose(float(computed_answer), float(llm_answer), rel_tol=1e-9):
                    logger.info(f"Consistent answers detected for row {index}: Computed: {computed_answer}, LLM: {llm_answer}")
                    consistent = True
                else:
                    logger.info(f"Inconsistent answers detected for row {index}: Computed: {computed_answer}, LLM: {llm_answer}")
                    consistent = False
            except:
                logger.info(f"Inconsistent answers detected for row {index}: Computed: {computed_answer}, LLM: {llm_answer}")
                consistent = False
                # break
            if consistent:
                task["answer"] = computed_answer
                task["consistency"] = 1
                tasks_filterd.append(task)
                filtered_count+=1
            else:
                task["consistency"] = 0
                tasks_filterd.append(task)
            original_count+=1

            print ("*********************")
            print (f"{original_count} -> {filtered_count}")
            print ("*********************\n")

        elem["tasks"] = tasks_filterd
        high_quality_rows.append(elem)

    # high_quality_df.to_csv(csv_path, index=False)
    save_json_file(save_path, high_quality_rows)

    return high_quality_rows


if __name__ == "__main__":
    args = parse_args()

    # Load the input data
    if os.path.exists(args.input_file):
        data_list = load_json_file(args.input_file)
    else:
        logger.error(f"Input file not found: {args.input_file}")
        exit()

    # Initialize the OpenAI client wrapper with the given model name
    config = {
        "model_name": args.model_name,
    }
    llm = OpenAIClientWrapper(config)
    
    # Perform quality control on the loaded data
    name_mapping = {
        'sssd': 'SingleStepSingleDocumentTask',
        '2ssd': 'TwoStepSingleDocumentTask',
        '3ssd': 'ThreeStepSingleDocumentTask',
        'ss2d': 'SingleStepTwoDocumentTask',
        '2s2d': 'TwoStepTwoDocumentTask',
        '3s2d': 'ThreeStepTwoDocumentTask',
        '3s3d': 'ThreeStepThreeDocumentTask'
    }

    high_quality_df = quality_control(
        llm=llm,
        data_list=data_list,
        name=name_mapping[args.task_name],
        quality_control_flag=args.quality_control_flag,
        save_path=args.output_file
    )

    logger.info(f"High-quality questions saved to {args.output_file}")