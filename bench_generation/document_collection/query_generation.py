from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from bench_generation.utils.tools import extract_json_from_string, load_or_generate, load_json_file
from bench_generation.utils.openai_models import OpenAIClientWrapper
import argparse
import logging
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Subtopic and Query Generation using LLMs.")
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of the model to use.')
    parser.add_argument('--num_subtopics', type=int, default=3, help='Number of subtopics to generate per main topic.')
    parser.add_argument('--num_queries', type=int, default=5, help='Number of queries to generate per subtopic.')
    parser.add_argument('--time_period', type=str, default='March-2024-to-September-2024', help='Time range for the queries.')
    parser.add_argument('--llm_batch_size', type=int, default=5, help='Batch size for the LLM.')
    parser.add_argument('--save_path', type=str, default='./outputs/data/', help='Path to save data.')
    parser.add_argument('--generate_subtopics_flag', action='store_true', help='Generate subtopics and queries if set, otherwise load from file.')
    parser.add_argument('--main_topics_file', type=str, default='./outputs/data/main_topics.json', help='Path to the main topics file.')
    return parser.parse_args()


class Query(BaseModel):
    decomposable_query: str = Field(
        description="A query that can be divided into multiple sub-queries to reach an answer. It often requires multiple steps or pieces of information to fully resolve."
    )
    atomic_queries: List[str] = Field(
        description="A list of atomic sub-queries derived from the decomposable query. Each atomic query should contain one realistic object, intended for searching a specific piece of information on a website. These queries are indivisible and represent a single, straightforward question."
    )

# Pydantic model for subtopic and query generation output
class SubtopicAndQueryGeneration(BaseModel):
    subtopic_and_query_map: Dict[str, List[Dict[str, List[Query]]]] = Field(
        description="A dictionary where each key is a main topic and its value is a list of dictionaries, each containing a 'subtopic' and a list of 'Queries' ."
    )

    @classmethod
    def get_prompt_template(cls) -> str:
        return """You are tasked with generating subtopics and corresponding queries for a benchmark designed to evaluate large language models' abilities in mathematical and numerical reasoning within real-world scenarios. Your objective is to create subtopics and queries that provide complex reasoning tasks, enabling models to demonstrate numerical analysis and step-by-step reasoning.

Instructions:

1. For Each Main Topic Provided:
  - Generate {num_subtopics} relevant subtopics.
  - Ensure each subtopic is challenging and involves complex reasoning and numerical data manipulation.

2. For Each Subtopic:
  - Generate {num_queries} detailed queries requiring mathematical and numerical reasoning.
  - Structure each query into two parts:
    - Decomposable Query: A high-level, complex query that requires reasoning across multiple steps to resolve.
    - Atomic Queries: A list of simpler, indivisible sub-queries, each seeking a specific piece of data or a straightforward answer. The atomic queries should be individually answerable and, when combined, provide the information needed to resolve the decomposable query.

3. Ensure each query specifies relevant entities and the time period {time_period} within which numerical data should be gathered and reasoned upon.

Example Structure:

Topic: Financial Market Analysis

- Subtopic: Trends in Stock Prices
  - Decomposable Query: What was the overall percentage change in Nvidia's stock price from May 2024 to August 2024, and how did its volatility compare to Tesla's over the same period?
  - Atomic Queries:
    - Query 1: What was Nvidia's stock price in May 2024?
    - Query 2: What was Nvidia's stock price in August 2024?
    - Query 3: What was Tesla's stock price volatility in May 2024?
    - Query 4: What was Tesla's stock price volatility in August 2024?

Each decomposable query and its atomic queries should reflect realistic, complex scenarios involving numerical reasoning that align with the specified time period {time_period}. This design will facilitate rigorous testing of large language models’ abilities to process multi-step numerical reasoning tasks.

Ensure the generated subtopics include Trends in Stock Prices.

Input format:
Main topic: {main_topic}

Please follow the provided format and ensure the output aligns with the example queries for consistency and relevance. Use the following format for outputs:

{format_instructions}"""

    @classmethod
    def generate(cls, llm, topics, num_subtopics, num_queries, time_period):
        parser = PydanticOutputParser(pydantic_object=SubtopicAndQueryGeneration)
        prompt_template = PromptTemplate(
            template=cls.get_prompt_template(),
            input_variables=["num_subtopics", "num_queries", "time_period", "main_topic", "format_instructions"]
        )
        # Prepare the input format for prompt
        messages_list = []
        for topic in topics:
            input_x = prompt_template.format(
                num_subtopics=num_subtopics,
                num_queries=num_queries,
                time_period=time_period,
                main_topic=topic,
                format_instructions=parser.get_format_instructions()
            )
            # print ('----')
            # print (input_x)
            # print ('----')
            # assert 1==0
            messages_list.append([{"role": "user", "content": input_x}])
        
        # Call LLM API in parallel
        responses = llm.call_llm_api_parallel(messages_list, temperature=0, max_tokens=4096)
        response_json_dict = {}
        for response in responses:
            response_json = extract_json_from_string(response)
            if "subtopic_and_query_map" in response_json:
                response_json_dict.update(response_json["subtopic_and_query_map"])
        return response_json_dict

if __name__ == "__main__":
    args = parse_args()
    
    # Load main topics from file
    if os.path.exists(args.main_topics_file):
        topics = load_json_file(args.main_topics_file)
    else:
        logging.error(f"Main topics file not found: {args.main_topics_file}")
        topics = []

    if not topics:
        logging.error("No main topics found to generate subtopics and queries.")
        exit()

    # Initialize the OpenAI client wrapper with config
    config = {
        "model_name": args.model_name,
        "llm_batch_size": args.llm_batch_size,
    }
    
    llm = OpenAIClientWrapper(config)
    
    save_path = os.path.join(args.save_path, args.time_period)
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, 'subtopics_and_queries.json')
    
    time_period = ' '.join(args.time_period.split('-'))
    logging.info(f"time_period: {time_period}")
    # Load or generate subtopics and queries
    subtopic_and_query_map = load_or_generate(
        filename=filename,
        condition=args.generate_subtopics_flag,
        generate_func=SubtopicAndQueryGeneration.generate,
        llm=llm,
        topics=topics,
        num_subtopics=args.num_subtopics,
        num_queries=args.num_queries,
        time_period=time_period
    )
    
    logging.info("*** SubtopicAndQueryGeneration done")
    # logging.info(f"Generated subtopics and queries: {subtopic_and_query_map}")
