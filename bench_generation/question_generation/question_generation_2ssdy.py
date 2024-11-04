import argparse
import json
import logging
import os
from typing import Any, List

import pandas as pd
from pydantic import BaseModel, Field

from bench_generation.question_generation.question_generation_sssd import (
    QuantityCell,
    ReasoningTask,
    TaskGenerationBase,
    PromptAlignmentCheck,
)
from bench_generation.utils.openai_models import OpenAIClientWrapper
from bench_generation.utils.tools import extract_json_from_string
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for the task generation script."""
    parser = argparse.ArgumentParser(
        description="Two-Step Single-Document Task Generation using LLMs."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt-4o',
        help='Name of the model to use.'
    )
    parser.add_argument(
        '--llm_batch_size',
        type=int,
        default=5,
        help='Batch size for the LLM.'
    )
    parser.add_argument(
        '--generate_questions_flag',
        action='store_true',
        help='Generate tasks if set, otherwise load from file.'
    )
    parser.add_argument(
        '--summarized_documents_file',
        type=str,
        default='./outputs/data/March-2024-to-September-2024/summarized_documents.csv',
        help='Path to the summarized documents file.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./outputs/data/March-2024-to-September-2024/two_step_tasks.csv',
        help='Path to save the generated tasks.'
    )
    return parser.parse_args()


class ReasoningTaskList(BaseModel):
    quantity_cells: List[QuantityCell] = Field(
        description=(
            "A collection of QuantityCells that represent the extracted numerical information, "
            "relevant objects, their attributes, and any associated dates or locations from the "
            "document. This field serves as the basis for generating the question and its "
            "corresponding solution."
        )
    )
    tasks: List[ReasoningTask] = Field(
        description=(
            "A list of ReasoningTask elements, where each entry contains 'quantity_cells', "
            "'question', 'solution', and 'answer'. The list should consist of at least 10 "
            "different ReasoningTask elements, if supported by the document(s), each evaluating "
            "different aspects of the model's numerical reasoning capabilities."
        )
    )

class TwoStepSingleDocumentTask(TaskGenerationBase):
    INSTRUCTION_TEMPLATE = """
Your task is to generate a real-world numerical reasoning question based on the information within a single document. The question should involve exactly two arithmetic operations (e.g., + followed by *, or - followed by /) and be solvable using basic math. Follow the steps below carefully:

Instructions:
1. Extract Quantity Cells: Identify all relevant numerical details from the document, such as objects, attributes, numerical values, and any related information (e.g., dates, locations, quantities, prices, or measurements). **Include specific time periods (e.g., May 2024) or locations (e.g., New York Stock Exchange) to help distinguish between different instances of the object.**
2. Generate a Real-World Question: Using the extracted information, create a question that can be solved by two arithmetic operations (e.g., +, -, *, /). The question must reflect a real-world scenario and **include specific time or location details to help identify the relevant object from the document**. Avoid directly mentioning the numerical values from the document. Ensure the model must infer and calculate the solution based on these values.
3. Write a Python Solution: Develop a Python function that solves the generated question using two basic arithmetic steps. The solution must:
    - Be executable in Python.
    - Avoid using function arguments; assign numerical values to variables directly.
    - Store the result in a variable named `answer` and return it.
4. Present the Final Answer: The final answer must be a single Arabic numeral.

Example Python function format:
```python
def solve():
    # Extracted numerical values from the document
    variable_1 = numerical_value_1  # Attribute of Object 1 at a specific time or location
    variable_2 = numerical_value_2  # Attribute of Object 2 at a different time or location
    
    # Perform two arithmetic operations (e.g., addition followed by multiplication)
    intermediate_result = variable_1 + variable_2
    answer = intermediate_result * variable_3  # Replace with actual operations required for the solution
    return answer
```

Input:
- Document: {document}
Output:
- {format_instructions}
"""

    ALIGNMENT_INSTRUCTION = """Real-World Question Creation:
Construct a question using the extracted information that requires exactly two arithmetic operations (choose from addition, subtraction, multiplication, or division) for solving.
Embed the question within a realistic scenario, adding specific time or location details as needed to uniquely identify the relevant object in the document.
Do not directly include numerical values from quantity cells or the document; instead, design the question so that the model must infer and perform the calculation based on those values.
Ensure that the question can be solved in exactly two arithmetic steps."""

    ALIGNMENT_PROMPT_TEMPLATE = """Do you think the following generated question and solution align with the instruction? If not, answer "No" and explain briefly why it does not align. If it aligns, answer "Yes." \n\nGenerated Question and Answer: {response} \n\nInstruction: {instruction} \n\nOutput Format:{format_instructions}"""

    REVISION_PROMPT_TEMPLATE = """Based on the provided feedback, revise the generated question from the previous response for enhanced clarity and accuracy. The output should be structured as a JSON dictionary with the following keys:
- relevant_quantity_cells: the specific information or quantities needed,
- question: containing the rephrased question,
- solution: detailing the method or steps to answer the question,
- answer: with the final answer.
Ensure that the rephrased question accurately conveys the original intent and detail.

Document:
{document}

Previous Response:
{response}

Feedback:
{feedback}

New Response (in Json):
{format_instructions}
"""

    @classmethod
    def get_prompt_template(cls) -> str:
        return cls.INSTRUCTION_TEMPLATE

    @classmethod
    def get_input_variables(cls) -> List[str]:
        return ["document", "format_instructions"]

    @classmethod
    def generate(cls, llm: Any, **kwargs) -> pd.DataFrame:
        parser = PydanticOutputParser(pydantic_object=ReasoningTaskList)
        prompt_template = PromptTemplate(
            template=cls.get_prompt_template(),
            input_variables=cls.get_input_variables()
        )

        messages_list = cls.prepare_messages_list(kwargs, prompt_template, parser)
        responses = []
        total_tasks_generated = 0
        total_tasks_successful = 0

        for idx, message_list in enumerate(messages_list):
            response = llm.call_llm_api(
                message_list, temperature=0.7, max_tokens=4096
            )
            try:
                response_json = extract_json_from_string(response)
                tasks = response_json.get('tasks', [])
            except (json.JSONDecodeError, KeyError):
                logger.error(f"Failed to parse response at index {idx}.")
                responses.append({})
                continue

            new_tasks = cls.filter_and_revise_tasks(
                llm, tasks, kwargs.get('document', ''), cls.ALIGNMENT_INSTRUCTION
            )

            total_tasks_generated += len(tasks)
            total_tasks_successful += len(new_tasks)
            logger.info(
                f"Document {idx}: {len(new_tasks)} out of {len(tasks)} tasks passed alignment."
            )

            response_dict = {
                "quantity_cells": response_json.get("quantity_cells", []),
                "tasks": new_tasks
            }
            responses.append(response_dict)

        logger.info(
            f"Final: {total_tasks_successful} tasks passed alignment "
            f"out of {total_tasks_generated} generated tasks."
        )

        df = kwargs['dataframe']
        if len(responses) != len(df):
            logger.error("Mismatch between responses and dataframe length.")
        df[cls.__name__] = responses
        df[cls.__name__] = df[cls.__name__].apply(json.dumps)
        return df

    @classmethod
    def filter_and_revise_tasks(cls, llm, tasks, document, instruction):
        align_parser = PydanticOutputParser(pydantic_object=PromptAlignmentCheck)
        new_tasks = []

        for task in tasks:
            alignment_passed, explanation = cls.check_alignment(
                llm, task, instruction, align_parser
            )
            if alignment_passed:
                new_tasks.append(task)
                logger.debug(f"Task aligned: {task['question']}")
            else:
                revised_task = cls.revise_task(
                    llm, task, document, explanation
                )
                if revised_task:
                    alignment_passed, _ = cls.check_alignment(
                        llm, revised_task, instruction, align_parser
                    )
                    if alignment_passed:
                        new_tasks.append(revised_task)
                        logger.debug(
                            f"Revised task aligned: {revised_task['question']}"
                        )
        return new_tasks

    @classmethod
    def check_alignment(cls, llm, task, instruction, parser):
        align_prompt_template = PromptTemplate(
            template=cls.ALIGNMENT_PROMPT_TEMPLATE,
            input_variables=["response", "instruction", "format_instructions"],
        )
        align_message = [{
            "role": "user",
            "content": align_prompt_template.format(
                response=json.dumps(task),
                instruction=instruction,
                format_instructions=parser.get_format_instructions()
            )
        }]
        align_response = llm.call_llm_api(
            align_message, temperature=0.0, max_tokens=4096
        )
        try:
            align_response_json = extract_json_from_string(align_response)
            alignment = align_response_json.get('alignment', '')
            explanation = align_response_json.get('explanation', '')
            return alignment == 'Yes', explanation
        except (json.JSONDecodeError, KeyError):
            logger.error("Failed to parse alignment response.")
            return False, ''

    @classmethod
    def revise_task(cls, llm, task, document, feedback):
        revision_parser = PydanticOutputParser(pydantic_object=ReasoningTask)
        revision_prompt_template = PromptTemplate(
            template=cls.REVISION_PROMPT_TEMPLATE,
            input_variables=["document", "response", "feedback", "format_instructions"],
        )
        revision_message = [{
            "role": "user",
            "content": revision_prompt_template.format(
                document=document,
                response=json.dumps(task),
                feedback=feedback,
                format_instructions=revision_parser.get_format_instructions()
            )
        }]
        revision_response = llm.call_llm_api(
            revision_message, temperature=0.0, max_tokens=4096
        )
        try:
            revised_task = extract_json_from_string(revision_response)
            return revised_task
        except json.JSONDecodeError:
            logger.error("Failed to parse revised task.")
            return None


if __name__ == "__main__":
    args = parse_args()

    # Load summarized documents from file
    logging.info(f"Task generation starts")
    if os.path.exists(args.summarized_documents_file):
        summarized_df = pd.read_csv(args.summarized_documents_file)
    else:
        logging.error(f"Summarized documents file not found: {args.summarized_documents_file}")
        summarized_df = pd.DataFrame()

    if summarized_df.empty:
        logging.error("No summarized documents found for task generation.")
        exit()

    # Initialize the OpenAI client wrapper with config
    config = {
        "model_name": args.model_name,
        "llm_batch_size": args.llm_batch_size,
    }
    
    llm = OpenAIClientWrapper(config)
    
    # Load or generate two-step single-document tasks
    tasks_df = TwoStepSingleDocumentTask.run(
        llm=llm,
        dataframe=summarized_df[:],
        csv_path=args.output_file,
        generate_questions_flag=args.generate_questions_flag
    )

    if 'Document_ID' not in tasks_df.columns:
        tasks_df['Document_ID'] = 'DOC_' + tasks_df.index.astype(str)

    # Save generated tasks to CSV
    assert len(tasks_df) == len(summarized_df)
    tasks_df.to_csv(args.output_file, index=False)
    logging.info(f"Generated tasks saved to {args.output_file}")