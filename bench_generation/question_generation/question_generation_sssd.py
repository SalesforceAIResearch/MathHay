import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from bench_generation.utils.tools import extract_json_from_string, load_or_generate, load_json_file, save_json_file
from bench_generation.utils.openai_models import OpenAIClientWrapper
import pandas as pd
import argparse
import logging
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Single-Step Single-Document Task Generation using LLMs.")
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of the model to use.')
    parser.add_argument('--llm_batch_size', type=int, default=5, help='Batch size for the LLM.')
    parser.add_argument('--generate_questions_flag', action='store_true', help='Generate tasks if set, otherwise load from file.')
    parser.add_argument('--summarized_documents_file', type=str, default='./outputs/data/March-2024-to-September-2024/summarized_documents.csv', help='Path to the summarized documents file.')
    parser.add_argument('--output_file', type=str, default='./outputs/data/March-2024-to-September-2024/single_step_tasks.csv', help='Path to save the generated tasks.')
    return parser.parse_args()


class QuantityCell(BaseModel):
    object: str = Field(
        description="The object related to the quantity (e.g., Tesla's stock)."
    )
    numerical_value: float = Field(
        description="The numerical value associated with the object (e.g., the stock price)."
    )
    date: str = Field(
        description="The specific date related to the quantity (e.g., May 2024)."
    )
    location: str = Field(
        description="The specific location related to the quantity, if applicable (e.g., New York Stock Exchange)."
    )
    context: str = Field(
        description="Any additional context or background information that helps distinguish this quantity from others."
    )

class ReasoningTask(BaseModel):
    relevant_quantity_cells: List[QuantityCell] = Field(
        description="A collection of QuantityCells that serves as the basis for generating the question and its corresponding solution. The QuantityCells include specific object, numerical values, and associated context (like date and location) to distinguish them."
    )
    question: str = Field(
        description="A factual question generated from a subset of the QuantityCells. The question should reference the specific time, location, or context to help identify the relevant object and challenge the model to reason about the correct quantity."
    )
    solution: str = Field(
        description="A Python function that solves the generated question using basic arithmetic operations. The solution must be executable, with clearly named variables reflecting the extracted information and a result assigned to a variable named `answer`. The solution demonstrates the reasoning process leading to the final answer."
    )
    steps: int = Field(
        description="How many operations(+, -, *, /), i.e., computational steps in python solution."
    )
    answer: float = Field(
        description="The final numerical answer to the question, presented as an Arabic numeral. This value is computed by the Python solution and represents the correct outcome of the reasoning task."
    )

# Define the list of reasoning tasks
class ReasoningTaskList(BaseModel):
    quantity_cells: List[QuantityCell] = Field(
        description="A collection of QuantityCells that represent the extracted numerical information, relevant objects, their attributes, and any associated dates or locations from the document. This field serves as the basis for generating the question and its corresponding solution."
    )
    tasks: List[ReasoningTask] = Field(
        description="A list of ReasoningTask elements, where each entry contains 'quantity_cells', 'question', 'solution', and 'answer'. The list should consist of at least 2 different ReasoningTask elements, if supported by the document(s), each evaluating different aspects of the model's numerical reasoning capabilities."
    )

class PromptAlignmentCheck(BaseModel):
    alignment: str = Field(
        description="Indicates if the generated question and solution align with the given instruction. It can be either 'Yes' if it aligns or 'No' if it does not."
    )
    explanation: str = Field(
        description="A brief explanation detailing why the generated content does or does not align with the instruction."
    )

# Example for Single-Step Single-Document Task
class SingleStepSingleDocumentTask():

    @classmethod
    def get_single_step_single_document_task_prompt_template(cls) -> str:
        return """
Your task is to generate a real-world numerical reasoning question based on the information contained within a single document. The question should involve a single arithmetic operation (+, -, *, /) and be solvable using basic math. Follow the steps below carefully:

Instructions:
1. Extract Quantity Cells: Identify all relevant numerical details from the document, such as objects, attributes, numerical values, and any related information (e.g., dates, locations, quantities, prices, or measurements). **Be sure to include specific time periods (e.g., May 2024) or locations (e.g., New York Stock Exchange) to help distinguish between different instances of the object.**
2. Generate a Real-World Question: Using the extracted information, create a question that can be solved by a single arithmetic operation (one of: +, -, *, /). The question must reflect a real-world scenario and **include specific time or location details to help identify the relevant object from the document**. Avoid directly mentioning the numerical values from the document. Instead, ensure the model must infer and calculate the solution based on these values.
3. Write a Python Solution: Develop a Python function that solves the generated question using basic arithmetic. The solution must:
    - Be executable in Python.
    - Avoid using function arguments; instead, assign numerical values to variables directly.
    - Ensure the result is stored in a variable named answer and returned by the function.
4. Present the Final Answer: The final answer must be a single Arabic numeral.

Example Python function format:
```python
def solve():
    # Extracted numerical values from the document
    variable_1 = numerical_value_1  # Attribute of Object 1 at a specific time or location
    variable_2 = numerical_value_2  # Attribute of Object 2 at a different time or location
    
    # Perform the arithmetic operation (e.g., addition)
    answer = variable_1 + variable_2  # Replace with the actual operation required for the solution
    return answer
```

Input:
- Document: {document}
Output:
- {format_instructions}
"""

    @classmethod
    def check_alignment(cls, llm, task: ReasoningTask) -> PromptAlignmentCheck:
        # This function uses LLM to verify if the generated question aligns with instructions
        alignparser = PydanticOutputParser(pydantic_object=PromptAlignmentCheck)
        alignment_check_prompt = f"""
        Check if the following question and solution align with the given instruction.
        Question: {task.question}
        Solution: {task.solution}
        Instruction: 
        - Using the extracted information from quantity cells
        - Questions can be solved by **one** single arithmetic operations (+, -, *, /). 
        - The question must reflect a real-world scenario and **include specific time or location details to help identify the relevant object from the document**. 
        - Avoid directly mentioning the numerical values from the document. Instead, ensure the model must infer and calculate the solution based on these values.

        Answer with 'Yes' or 'No' for alignment and provide a brief explanation.

        {alignparser.get_format_instructions()}
        """
        
        response = llm.call_llm_api([{"role": "user", "content": alignment_check_prompt}], temperature=0.7)
        try:
            response_json = extract_json_from_string(response)
            return PromptAlignmentCheck.parse_obj(response_json)
        except:
            return None
        

    @classmethod
    def refine_task(cls, llm, task: ReasoningTask, feedback: str, cells: List[QuantityCell]) -> ReasoningTask:
        # This function uses the feedback to refine the task
        reasoningTaskParser = PydanticOutputParser(pydantic_object=ReasoningTask)
        refinement_prompt = f"""
        Given the following feedback, refine the question and solution accordingly.
        
        All relevant quantity cells: {cells}
        Preivous Relevant Quantity Cells: {task.relevant_quantity_cells}
        Preivous Question: {task.question}
        Preivous Solution: {task.solution}

        Feedback: {feedback}
        
        Provide the refined question and solution.
        {reasoningTaskParser.get_format_instructions()}
        """
        
        response = llm.call_llm_api([{"role": "user", "content": refinement_prompt}], temperature=0.7)
        try:
            response_json = extract_json_from_string(response)
            return ReasoningTask.parse_obj(response_json)
        except:
            return None
        

    @classmethod
    def generate(cls, llm: Any, **kwargs) -> pd.DataFrame:
        ReasoningTaskListParser = PydanticOutputParser(pydantic_object=ReasoningTaskList)
        SingleStepSingleDocumentTaskPromptTemplate = PromptTemplate(
            template=cls.get_single_step_single_document_task_prompt_template(),
            input_variables=["document", "format_instructions"]
        )
        # messages_list = cls.prepare_messages_list(kwargs, prompt_template, parser)

        data_stat = {"total":0, "orign_correct":0, 'refined_correct':0}
        data_list = kwargs["data_list"]
        data_examples = []
        for elem in data_list:
            filtered_documents = elem["filtered_documents"]
            doc_ids = elem["doc_ids"]
            for doc_idx in range(len(doc_ids)):
                doc_text = filtered_documents[doc_idx]
                doc_id = doc_ids[doc_idx]

                message_list = [{"role": "user", "content": SingleStepSingleDocumentTaskPromptTemplate.format(
                    document=doc_text,
                    format_instructions=ReasoningTaskListParser.get_format_instructions()
                )}]

                response = llm.call_llm_api(message_list, temperature=0.7, max_tokens=4096)

                try:
                    response_json = extract_json_from_string(response)
                except:
                    response_json = {}
                
                try:
                    taskList = ReasoningTaskList.parse_obj(response_json)
                except:
                    continue
                taskList_dict = taskList.dict()
                # print ('taskList_dict', taskList_dict.keys())
                # task = ReasoningTask.parse_obj(response)
                new_tasks = []
                # print (taskList)
                # assert 1==0
                for task in taskList.tasks:
                    # Alignment check and refinement loop
                    success_flag = 0
                    data_stat['total']+=1
                    for _ in range(3):  # Max 3 attempts
                        alignment_check = cls.check_alignment(llm, task)
                        if alignment_check is None:
                            continue
                        if alignment_check.alignment == "Yes":
                            success_flag = 1
                            new_task = task
                            data_stat['orign_correct']+=1
                            break  # Stop if aligned
                        else:
                            # Refine based on feedback
                            task = cls.refine_task(llm, task, alignment_check.explanation, taskList.quantity_cells)
                            # task = new_task

                    if success_flag:
                        new_tasks.append(new_task.dict())
                        if _ >0:
                            data_stat['refined_correct']+=1
                    print (f"success_flag: {success_flag}")
                taskList_dict['tasks'] = new_tasks

                if new_tasks != []:
                    data_example = {
                        "Topic": elem["Topic"],
                        "Subtopic": elem["Subtopic"],
                        "decomposable_query": elem["decomposable_query"],
                        "atomic_queries": elem["atomic_queries"],
                        "doc_id": doc_id,
                        "document": doc_text,
                        "tasks": new_tasks
                    }
                    data_examples.append(data_example)
            
                logging.info(f"data sta: {data_stat}")
        
        return data_examples



    @classmethod
    def run(cls, llm: Any, data_list: Dict, dave_path: str, generate_questions_flag: bool) -> pd.DataFrame:
        """
        Run the task generation or load from Json based on the generate_questions_flag.
        
        :param llm: The language model client wrapper.
        :param dataframe: The input DataFrame containing necessary columns.
        :param json_path: The file path to save or load the Json.
        :param generate_questions_flag: A boolean flag; if True, generate tasks and save as Json. Otherwise, loadJson.
        :return: The DataFrame with generated tasks or loaded data.
        """
        if generate_questions_flag:
            logger.info("Generating tasks and saving to CSV.")
            data_result = cls.generate(llm, data_list=data_list)
            save_json_file(dave_path, data_result)
            return data_result
        else:
            logger.info("Loading tasks from existing CSV.")
            data_result = load_json_file(dave_path)
            return data_result
            # sss

if __name__ == "__main__":
    args = parse_args()
    
    # Load summarized documents from file
    logging.info(f"Task generation starts")

    summarized_data = load_json_file(args.summarized_documents_file)

    # Initialize the OpenAI client wrapper with config
    config = {
        "model_name": args.model_name,
        "llm_batch_size": args.llm_batch_size,
    }
    
    llm = OpenAIClientWrapper(config)
    
    # Load or generate single-step single-document tasks
    tasks_data = SingleStepSingleDocumentTask.run(
        llm=llm,
        data_list=summarized_data,
        dave_path=args.output_file,
        generate_questions_flag=args.generate_questions_flag
    )

    # save_json_file(tasks_data, args.output_file)
    logging.info(f"Generated tasks saved to {args.output_file}")
