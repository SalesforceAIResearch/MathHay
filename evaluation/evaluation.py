import pandas as pd
import json
import os
import math
from pydantic import BaseModel, Field
from typing import List, Tuple
from bench_generation.utils.openai_models import OpenAIClientWrapper
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import logging
import argparse
import google
from bench_generation.utils.tools import extract_json_from_string, load_jsonl_file, load_json_file, save_json_file, save_jsonl_file
import time
import boto3
from botocore.exceptions import ClientError
import tiktoken

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the tiktoken encoder
encoding = tiktoken.encoding_for_model("gpt-4o")

def tokenization(text):
    """Calculate the number of tokens in the text using tiktoken."""
    return encoding.encode(text, disallowed_special=())

def decode_tokens(tokens):
    """Decode a list of tokens back into text using tiktoken."""
    return encoding.decode(tokens)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate or load data for benchmarking.")
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of the model to use.')
    parser.add_argument('--llm_batch_size', type=int, default=5, help='Batch size for the LLM.')
    parser.add_argument('--task', type=str, default='threes3d', help='Task to choose.')
    parser.add_argument('--haystack_len', type=str, default="32000", help='Choose from "32000", "64000", "128000".')
    parser.add_argument('--placement', type=str, default="middle", help='Choose from first, middle, last, p1, p2, p3.')
    parser.add_argument('--mode', type=str, default='verified', help='Model mode to choose: gpt-4o, gemini, claude, qwen.')
    parser.add_argument('--input_dir', type=str, default='./outputs/data/March-2024-to-September-2024/', help='') #summarized_documents.csv
    parser.add_argument('--output_dir', type=str, default='./outputs/results/March-2024-to-September-2024/', help='') #summarized_documents.csv
    parser.add_argument('--input_file', type=str, default='verified_haystack_question_2s2d.json',
                        help='Path to the file containing generated questions.')
    
    return parser.parse_args()


def gemini_inference(llm, prompt, max_retries=5, backoff_factor=1):
    """Generates content with retry logic and backoff in case of errors."""
    retries = 0
    while retries < max_retries:
        try:
            response = llm.generate_content(prompt).text
            return response
        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Error: {e}. Retrying in {backoff_factor} seconds...")
            retries += 1
            time.sleep(backoff_factor)
            backoff_factor *= 2  # Exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None  # Return None if retries are exhausted

class QuantityCell(BaseModel):
    quantity_cell: Tuple[str] = Field(
        description="A tuple containing details about a specific object, including the nouns of the object, its attributes, numerical values, relevant dates, and locations. This cell encapsulates all information required for extracting and computing the answer to the reasoning question."
    )

# Define the Pydantic model for reasoning and answer verification
class ReasoningVerification(BaseModel):
    reasoning: str = Field(description="Solution process.")
    answer: float = Field(description="The final numerical answer to the question, deduced through reasoning.")

# Define a Pydantic model for ReasoningVerification
class LLMVerification(BaseModel):
    reasoning: str = Field(description="Verification process.")
    output: str = Field(description="Yes or No. Yes means the two solutions are equivalent. No means the two solutions are different.")

def prompt_len_cal(llm, question) -> float:
    """
    Verify the answer by asking the LLM to solve the problem based on the Pydantic model task.
    
    :param llm: The language model client wrapper.
    :param question: The question to solve.
    :return: The answer generated by the LLM.
    """
    # Use the PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=ReasoningVerification)
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="""
Long-Context Documents:
{long_context_input}

You are tasked with solving a mathematical reasoning question using information from Long-Context Documents. Follow these steps to ensure accurate extraction and calculation:

**Instructions:**
1. **Extract Relevant Numerical Information**: Carefully read through the provided Long-Context Documents to identify and list all relevant numerical details. These could include objects, their attributes, numerical values, dates, locations, or any other quantitative data.
   
2. **Analyze and Solve the Question**: Use the identified numerical details to solve the given question. Ensure your solution involves a single computational step based on the relevant data extracted. Focus on logical or arithmetic operations as required by the question.

Question:
{question}

{format_instructions}

        """,
        input_variables=["question", "quantity_cells", "format_instructions"]
    )
    
    # Construct the prompt using the provided question and relevant quantity cells
    prompt = prompt_template.format(
        question=question,
        long_context_input="",
        format_instructions=parser.get_format_instructions()
    )
    return len(tokenization(prompt))

def initialize_llm(args):
    """Initialize the appropriate LLM based on the mode."""
    llm_config = {
        "model_name": args.model_name,
        "llm_batch_size": args.llm_batch_size,
    }

    verification_llm_config = {
        "model_name": "gpt-4o",
        "llm_batch_size": 10,
    }
    verification_llm = OpenAIClientWrapper(verification_llm_config)

    if args.model_name in ['gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview']:
        llm = OpenAIClientWrapper(llm_config)
    elif args.model_name == 'gemini-1.5-pro-002':
        import google.generativeai as genai
        genai.configure(api_key="AIzaSyDFXVjdCHcCk7hUiZhTv1EgrOluBFME5k0") #os.environ["GEMINI_API_KEY"]
        # llm = genai(model_path="gemini-1.5-pro")  # Replace with actual Gemini initialization
        llm = genai.GenerativeModel(model_name=args.model_name)
    elif args.model_name == 'claude':
        llm = boto3.client("bedrock-runtime", region_name="us-east-1")
        # import anthropic
        # llm = anthropic.Anthropic(api_key=args.api_key)  # Replace with actual Claude initialization
    elif args.model_name == 'qwen':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = 'Qwen/Qwen1.5-7B'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()
        llm.tokenizer = tokenizer
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    return llm, verification_llm


def compare_answers(computed_answer, llm_answer, index):
    # Compare answers and check consistency
    try:
        if computed_answer is None or llm_answer is None or math.isclose(float(computed_answer), float(llm_answer), rel_tol=1e-9):
            logger.info(f"Consistent answers detected for row {index}: Computed: {computed_answer}, LLM: {llm_answer}")
            return True
        else:
            logger.info(f"Inconsistent answers detected for row {index}: Computed: {computed_answer}, LLM: {llm_answer}")
            return False
    except:
        logger.info(f"Inconsistent answers detected for row {index}: Computed: {computed_answer}, LLM: {llm_answer}")
        return False

def llm_verification_func(llm, solution1, solution2) -> float:
    """
    Verify the answer by asking the LLM to solve the problem based on the Pydantic model task.
    
    :param llm: The language model client wrapper.
    :param question: The question to solve.
    :return: The answer generated by the LLM.
    """
    # Use the PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=LLMVerification)
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="""
Your task is to determine if the two given solutions are equivalent in terms of reasoning and final answer.

Solution 1:
{solution1}

Solution 2:
{solution2}

Criteria for equivalence:
1. Both solutions should have the same reasoning steps leading to the final answer.
2. The final numerical answers should be identical.

Please analyze the two solutions and state whether they are the same or different. If different, provide a brief explanation of the discrepancies.

Example:
Solution 1:
def solve():
    current_value = 45e9  # $45 billion
    projected_value = 400e9  # $400 billion
    answer = projected_value - current_value
    return answer
Answer1: 355000000000.0

Solution 2:
The current value of the AI chip market is projected to be $45 billion, and it is expected to rise to $400 billion by 2027. To find the difference, we subtract the current value from the projected value: $400 billion - $45 billion = $355 billion.
Answer2: 355.0

Output: Yes

{format_instructions}
        """,
        input_variables=["solution1", "solution2", "format_instructions"]
    )
    
    # Construct the prompt using the provided question and relevant quantity cells
    prompt = prompt_template.format(
        solution1=solution1,
        solution2=solution2,
        format_instructions=parser.get_format_instructions()
    )
    message_list = [{"role": "user", "content": prompt}]
    try:
        response = llm.call_llm_api(message_list, temperature=0, max_tokens=1024)
    except:
        response = None
 
    try:
        response_json = extract_json_from_string(response)
        verified_task = LLMVerification(**response_json)  # Parsing response into the Pydantic model
        return verified_task.output, verified_task.reasoning
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return "None", "None"

def llm_inference(llm, question, long_context_input, model_name) -> float:
    """Infer answer based on the selected LLM."""
    parser = PydanticOutputParser(pydantic_object=ReasoningVerification)
    prompt_template = PromptTemplate(
        template="""
Long-Context Documents:
{long_context_input}

You are tasked with solving a mathematical reasoning question using information from Long-Context Documents. Follow these steps to ensure accurate extraction and calculation:

**Instructions:**
1. **Extract Relevant Numerical Information**: Carefully read through the provided Long-Context Documents to identify and list all relevant numerical details. These could include objects, their attributes, numerical values, dates, locations, or any other quantitative data.
   
2. **Analyze and Solve the Question**: Use the identified numerical details to solve the given question. Ensure your solution involves a single computational step based on the relevant data extracted. Focus on logical or arithmetic operations as required by the question.

Question:
{question}

{format_instructions}
        """,
        input_variables=["question", "quantity_cells", "format_instructions"]
    )
    
    prompt = prompt_template.format(
        question=question,
        long_context_input=long_context_input,
        format_instructions=parser.get_format_instructions()
    )

    # message_list = [{"role": "user", "content": prompt}]
    # response = llm.call_llm_api(message_list, temperature=0, max_tokens=1024)
    # print (response)

    if model_name in ['gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview'] :
        message_list = [{"role": "user", "content": prompt}]
        # response = llm.call_llm_api(message_list, temperature=0, max_tokens=1024)
        # print (response)
        try:
            response = llm.call_llm_api(message_list, temperature=0, max_tokens=1024)
        except:
            logger.error(f"Error LLM response")
            return -float("inf"), "None"
    elif model_name == 'gemini-1.5-pro-002':
        # response = llm.generate_content(prompt) #generation_config=generation_config, stream=True
        # response = llm.generate_content(prompt).text
        response = gemini_inference(llm, prompt)
    elif model_name == 'claude':
        MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        message_list = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                ],
            }
        ]

        response = llm.converse(
            modelId=MODEL_ID,
            messages=message_list,
        )
        response = response["output"]["message"]["content"][0]["text"]
    elif model_name == 'qwen':
        inputs = llm.tokenizer(prompt, return_tensors='pt').to('cuda')
        outputs = llm.generate(**inputs, max_new_tokens=100)
        response = llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    try:
        response_json = extract_json_from_string(response)
        verified_task = ReasoningVerification(**response_json)
        return verified_task.answer, verified_task.reasoning
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return -float("inf"), "None"


def evaluation(args):
    logger.info(args)
    llm, verification_llm = initialize_llm(args)

    document_path = os.path.join(args.input_dir, 'documents.json')
    document_data = load_json_file(document_path)  # Load the summarization DataFrame
    load_path = os.path.join(args.input_dir, args.input_file)
    data_rows = load_json_file(load_path)
    parser = PydanticOutputParser(pydantic_object=ReasoningVerification)
    format_instructions = parser.get_format_instructions()
    save_path = os.path.join(args.output_dir, f"{args.mode}_{args.model_name}_{args.task}_{args.haystack_len}_{args.placement}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    
    consistent_count = 0
    total_count = 0
    for index, row in enumerate(data_rows):
        question = row["Task"]['question']
        expected_answer = row["Task"]["answer"]
        longest_irrelevant_documents_indexs = row["Irrelevant_Documents_Indexs"]
        longest_irrelevant_documents = [document_data[doc_idx]["Document"] for doc_idx in longest_irrelevant_documents_indexs]
        # longest_irrelevant_documents = summarization_df.loc[longest_irrelevant_documents_indexs, 'Document'].tolist()
        joint_longest_irrelevant_documents = "\n\n".join(longest_irrelevant_documents)
        joint_longest_irrelevant_documents_tokens = tokenization(joint_longest_irrelevant_documents)
        joint_longest_irrelevant_documents_len = len(joint_longest_irrelevant_documents_tokens)
        
        if args.task in ["sssd", "2ssd", "3ssd"]:
            relevant_document = row['Documents'][0]
            relevant_document_tokens = tokenization(relevant_document)
            prompt_len = prompt_len_cal(llm, question)
            set_haystack_len = int(args.haystack_len)
            if args.haystack_len == "128000":
                if args.model_name in ['o1-mini', 'o1-preview']:
                    # print ("dkjfskjfksdjfks")
                    set_haystack_len = set_haystack_len-20000
                else:
                    set_haystack_len = set_haystack_len-2000
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            if args.placement == 'first':
                placement_index = 0
            elif args.placement == 'middle':
                placement_index = rest_tokens//2
            elif args.placement == 'last':
                placement_index = rest_tokens
            else:
                placement_index = rest_tokens//10*int(args.placement[1:])
            
            # print (placement_index, rest_tokens)
            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index:placement_index] = relevant_document_tokens

        elif args.task in ["ss2d", "2s2d", "3s2d"]:
            relevant_document1 = row['Documents'][0]
            relevant_document2 = row['Documents'][1]
            relevant_document1_tokens = tokenization(relevant_document1)
            relevant_document2_tokens = tokenization(relevant_document2)
            relevant_document = relevant_document1 + relevant_document2
            relevant_document_tokens = tokenization(relevant_document)
            prompt_len = prompt_len_cal(llm, question)
            set_haystack_len = int(args.haystack_len)
            if args.haystack_len == "128000":
                if args.model_name in ['o1-mini', 'o1-preview']:
                    # print ("dkjfskjfksdjfks")
                    set_haystack_len = set_haystack_len-20000
                else:
                    set_haystack_len = set_haystack_len-2000
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            if args.placement == 'first-first':
                placement_index1 = 0
                placement_index2 = 0
            elif args.placement == 'middle-middle':
                placement_index1 = rest_tokens//2
                placement_index2 = rest_tokens//2
            elif args.placement == 'last-last':
                placement_index1 = rest_tokens
                placement_index2 = rest_tokens
            elif args.placement == 'first-middle':
                placement_index1 = 0
                placement_index2 = rest_tokens//2
            elif args.placement == 'middle-last':
                placement_index1 = rest_tokens//2
                placement_index2 = rest_tokens
            elif args.placement == 'first-last':
                placement_index1 = 0
                placement_index2 = rest_tokens
            else:
                # xxxx = args.placement.split('-')
                # placement_index1 = int(xxxx[0])
                # placement_index2 = int(xxxx[1])
                placement_index1 = rest_tokens//10*int(args.placement[1:])
                placement_index2 = rest_tokens//10*int(args.placement[1:])

            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index2:placement_index2] = relevant_document2_tokens
            irrelevant_document_tokens_[placement_index1:placement_index1] = relevant_document1_tokens

        elif args.task in ["3s3d"]:
            relevant_document1 = row['Documents'][0]
            relevant_document2 = row['Documents'][1]
            relevant_document3 = row['Documents'][2]
            relevant_document1_tokens = tokenization(relevant_document1)
            relevant_document2_tokens = tokenization(relevant_document2)
            relevant_document3_tokens = tokenization(relevant_document3)
            relevant_document = relevant_document1 + relevant_document2 + relevant_document3
            relevant_document_tokens = tokenization(relevant_document)
            prompt_len = prompt_len_cal(llm, question)
            set_haystack_len = int(args.haystack_len)
            if args.haystack_len == "128000":
                if args.model_name in ['o1-mini', 'o1-preview']:
                    # print ("dkjfskjfksdjfks")
                    set_haystack_len = set_haystack_len-20000
                else:
                    set_haystack_len = set_haystack_len-2000
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            if args.placement == 'first-first-first':
                placement_index1 = 0
                placement_index2 = 0
                placement_index3 = 0
            elif args.placement == 'middle-middle-middle':
                placement_index1 = rest_tokens//2
                placement_index2 = rest_tokens//2
                placement_index3 = rest_tokens//2
            elif args.placement == 'last-last-last':
                placement_index1 = rest_tokens
                placement_index2 = rest_tokens
                placement_index3 = rest_tokens
            elif args.placement == 'first-middle-last':
                placement_index1 = 0
                placement_index2 = rest_tokens//2
                placement_index3 = rest_tokens
            else:
                # xxxx = args.placement.split('-')
                # placement_index1 = int(xxxx[0])
                # placement_index2 = int(xxxx[1])
                # placement_index3 = int(xxxx[1])
                placement_index1 = rest_tokens//10*int(args.placement[1:])
                placement_index2 = rest_tokens//10*int(args.placement[1:])
                placement_index3 = rest_tokens//10*int(args.placement[1:])

            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index3:placement_index3] = relevant_document3_tokens
            irrelevant_document_tokens_[placement_index2:placement_index2] = relevant_document2_tokens
            irrelevant_document_tokens_[placement_index1:placement_index1] = relevant_document1_tokens


        final_document_tokens = irrelevant_document_tokens_[:]
        long_context_input = decode_tokens(final_document_tokens)
        llm_answer, reasoning = llm_inference(llm, question, long_context_input, args.model_name)
        solution1 = row["Task"]["solution"] + "\nAnswer1: " + str(expected_answer)
        solution2 = reasoning + "\nAnswer2: " + str(llm_answer)
        llm_verification, ver_reasoning = llm_verification_func(llm, solution1, solution2)

        consistent = compare_answers(expected_answer, llm_answer, index)
        if "yes" in llm_verification.lower() or consistent is True:
            consistent_count += 1

        total_count += 1
        accuracy = consistent_count * 1.0 / total_count if total_count > 0 else 0
        logger.info(f"Accuracy of LLM responses: {accuracy:.2%}")

        result = {
            "index": index,
            "relevant_documents": relevant_document,
            "question": question,
            "expected_answer": expected_answer,
            "pred_answer": llm_answer,
            "correct": consistent,
            "llm_annotate_reasoning": row["Task"]["solution"],
            "llm_judge": llm_verification,
            "pred_reasoning": reasoning,
            "judge_reasoning": ver_reasoning
        }
        results.append(result)

    save_json_file(save_path, results)


if __name__ == "__main__":
    args = parse_args()
    evaluation(args)