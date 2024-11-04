import re
from typing import List, Dict, Union
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
import argparse
import logging
import os
from pydantic import BaseModel, Field
from tqdm import tqdm

from bench_generation.utils.tavily_search_models import TavilyClientWrapper
from bench_generation.utils.tools import load_or_generate, save_json_file, load_json_file

# Load spaCy English model
# Ensure to install the spaCy model before running: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
spacy.load("en_core_web_sm")

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Document Collection using LLMs and Search Engines.")
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of the model to use.')
    parser.add_argument('--min_numbers', type=int, default=10, help='Minimum number of distinct numerical values.')
    parser.add_argument('--min_sentences', type=int, default=5, help='Minimum number of sentences.')
    parser.add_argument('--min_words', type=int, default=100, help='Minimum number of words.')
    parser.add_argument('--min_entities', type=int, default=5, help='Minimum number of named entities.')
    parser.add_argument('--llm_batch_size', type=int, default=5, help='Batch size for the LLM.')
    parser.add_argument('--save_path', type=str, default='./outputs/data/', help='Path to save data.')
    parser.add_argument('--generate_documents_flag', action='store_true', help='Generate documents if set, otherwise load from file.')
    parser.add_argument('--time_period', type=str, default='March-2024-to-September-2024', help='Time range for the queries.')
    # parser.add_argument('--tvly_api_key', type=str, default="", help="")
    return parser.parse_args()

# Pydantic model for document filtering output
class FilteredDocument(BaseModel):
    main_topic: str = Field(description="The main topic of the document.")
    subtopic: str = Field(description="The subtopic under the main topic.")
    query: str = Field(description="The query used to fetch the document.")
    text: str = Field(description="The filtered document text content.")

class DocumentCollector:
    def __init__(self, tavily_client_wrapper: TavilyClientWrapper):
        self.tavily_client_wrapper = tavily_client_wrapper
    
    def contains_sufficient_context(self, text: str, min_sentences: int, min_words: int, min_entities: int) -> bool:
        """Checks if the document contains sufficient context by counting sentences, words, and entities."""
        sentences = re.split(r'[.!?]', text)
        sentence_count = len([s for s in sentences if s.strip()])
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        entity_count = len(set(entities))

        has_sufficient_sentences = sentence_count >= min_sentences
        has_sufficient_words = word_count >= min_words
        has_sufficient_entities = entity_count >= min_entities

        return has_sufficient_sentences and has_sufficient_words and has_sufficient_entities

    def contains_sufficient_numerical_data(self, text: str, min_numbers: int) -> bool:
        """Checks if the document contains more than a specified number of distinct numerical values."""
        numbers = re.findall(
            r'\b(?!\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b)'  # Exclude date formats
            r'(?!\b\d{4}\b)'  # Exclude standalone years
            r'\d+(?:,\d{3})*(?:\.\d+)?%?\b', text)  # Capture numbers, decimals, and percentages
        distinct_numbers = set(numbers)
        return len(distinct_numbers) > min_numbers

    def filter_documents(self, documents: Dict[str, Dict[str, Dict[str, List[str]]]], 
                         min_numbers: int, min_sentences: int, min_words: int, min_entities: int) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Filters a nested dictionary of documents based on content and numerical data requirements."""
        filtered_documents = {}
        total = 0
        for main_topic, subtopics in documents.items():
            filtered_subtopics = {}
            for subtopic, queries in subtopics.items():
                filtered_queries = {}
                for query, texts in queries.items():
                    filtered_texts = [
                        text for text in texts 
                        if text and self.contains_sufficient_numerical_data(text, min_numbers) and
                        self.contains_sufficient_context(text, min_sentences, min_words, min_entities)
                    ]
                    if filtered_texts:
                        filtered_queries[query] = filtered_texts[:1]  # Only keep the first filtered document
                        total += len(filtered_texts)
                if filtered_queries:
                    filtered_subtopics[subtopic] = filtered_queries
            if filtered_subtopics:
                filtered_documents[main_topic] = filtered_subtopics
        logging.info(f"Number of filtered documents: {total}")
        return filtered_documents

    def fetch_page_content(self, query_dict: Dict[str, List[Dict[str, List[str]]]]) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Fetches page content for each query in the provided query dictionary using TavilyClientWrapper."""
        page_content_dict = {}
        topic_count = 0

        # for main_topic, subtopics in query_dict.items():
        for main_topic, subtopics in tqdm(query_dict.items(), desc="Processing fetch_page_content"):
            page_content_dict[main_topic] = {}  # Initialize dict for main topic
            for subtopic_info in subtopics:
                subtopic = subtopic_info['subtopic']
                queries = subtopic_info['queries']
                page_content_dict[main_topic][subtopic] = {}  # Initialize dict for subtopic
                for query in queries:
                    documents = self.tavily_client_wrapper.search(query)
                    page_content_dict[main_topic][subtopic][query] = documents if documents else []  # Avoid None
                    if documents:
                        topic_count += 1
                    time.sleep(1)  # Pause between queries to avoid rate limiting

        logging.info("Search completed.")
        logging.info(f"Total successful queries: {topic_count}")
        return page_content_dict

if __name__ == "__main__":
    import copy
    args = parse_args()

    # Load query dictionary from file
    query_dict_file = os.path.join('./outputs/data/', args.time_period, 'subtopics_and_queries.json')
    if os.path.exists(query_dict_file):
        query_dict = load_json_file(query_dict_file)
    else:
        logging.error(f"Query dictionary file not found: {query_dict_file}")
        query_dict = {}

    if not query_dict:
        logging.error("No queries found for document collection.")
        exit()

    # ##### for testing
    # query_dict_ = {}
    # for k, v in query_dict.items():
    #     query_dict_[k] = v[:1]
    #     break
    # query_dict = copy.deepcopy(query_dict_)
    
    # Initialize the Tavily client wrapper
    tavily_client = TavilyClientWrapper()

    # Create the document collector
    collector = DocumentCollector(tavily_client_wrapper=tavily_client)

    # Define save path
    save_path = os.path.join(args.save_path, args.time_period, 'collected_documents.json')
    os.makedirs(args.save_path, exist_ok=True)

    # Load or generate documents
    document_data = load_or_generate(
        filename=save_path,
        condition=args.generate_documents_flag,
        generate_func=collector.fetch_page_content,
        query_dict=query_dict
    )

    logging.info(f"Searched documents saved to {save_path}")
    
    # Filter the documents
    filtered_documents = collector.filter_documents(
        documents=document_data,
        min_numbers=args.min_numbers,
        min_sentences=args.min_sentences,
        min_words=args.min_words,
        min_entities=args.min_entities
    )
    
    # Save filtered documents to file
    filtered_documents_file = os.path.join(args.save_path, args.time_period, 'filtered_documents.json')
    save_json_file(filtered_documents_file, filtered_documents)
    logging.info(f"Filtered documents saved to {filtered_documents_file}")
