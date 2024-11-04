from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from bench_generation.utils.tools import extract_json_from_string, load_or_generate
from bench_generation.utils.openai_models import OpenAIClientWrapper
import argparse
import logging
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Topic Generation using LLMs.")
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of the model to use.')
    parser.add_argument('--num_topics', type=int, default=5, help='Number of main topics to generate.')
    parser.add_argument('--llm_batch_size', type=int, default=5, help='Batch size for the LLM.')
    parser.add_argument('--save_path', type=str, default='./outputs/data/', help='Path to save data.')
    parser.add_argument('--generate_topic_flag', action='store_true', help='Generate topics if set, otherwise load from file.')
    return parser.parse_args()

# Pydantic model for topic generation output
class TopicGeneration(BaseModel):
    topic_list: List[str] = Field(description="A Python list where each element is a string representing a single topic. The list should only contain the topics, without any additional information or descriptions. Each topic should be concise.")

    @classmethod
    def get_prompt_template(cls) -> str:
        return """
You are tasked with generating a diverse set of topics for a benchmark designed to evaluate large language models' abilities in mathematical and numerical reasoning within real-world scenarios. 
The goal is to create topics where documents will contain ample numerical data and rich contextual information that can support complex reasoning tasks. 
The topics should span various real-world domains where mathematical reasoning is often required.

For each main topic, ensure that there is potential for generating subtopics that involve mathematical reasoning with substantial numerical content. 
Please provide {num_topics} main topics that fit these criteria and briefly describe how each topic can support tasks involving mathematical reasoning and numerical analysis in realistic contexts.

Ensure the generated topics include Financial Market Analysis and Sports Performance Analytics.

{format_instructions}
"""

    @classmethod
    def generate(cls, llm, num_topics: int = 5) -> List[str]:
        """Generate a list of topics using the given LLM."""
        logging.info(f"Generating {num_topics} topics using the provided LLM.")
        
        parser = PydanticOutputParser(pydantic_object=TopicGeneration)
        prompt_template = PromptTemplate(
            template=cls.get_prompt_template(),
            input_variables=["num_topics", "format_instructions"]
        )
        input_prompt = prompt_template.format(num_topics=num_topics, format_instructions=parser.get_format_instructions())
        print ('----')
        print (input_prompt)
        print ('----')
        # assert 1==0
        message = [{"role": "user", "content": input_prompt}]
        
        try:
            response = llm.call_llm_api(message, temperature=0, max_tokens=512)
            response_json = extract_json_from_string(response)
            topics = response_json.get("topic_list", [])
            
            if not topics:
                logging.error("No topics were generated. Please check the LLM response and prompt.")
            return topics
        
        except Exception as e:
            logging.error(f"An error occurred during topic generation: {e}")
            return []

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the OpenAI client wrapper with config
    config = {
        "model_name": args.model_name,
        "llm_batch_size": args.llm_batch_size,
    }
    
    llm = OpenAIClientWrapper(config)
    
    os.makedirs(args.save_path, exist_ok=True) 
    filename = os.path.join(args.save_path, 'main_topics.json')
    main_topic_list = load_or_generate(
        filename=filename, # os.path.join(results_folder, filename)
        condition=args.generate_topic_flag,
        generate_func=TopicGeneration.generate,
        llm=llm,
        num_topics=args.num_topics
    )
    logging.info("*** TopicGeneration done")
    logging.info(f"Generated topics: {main_topic_list}")
    
