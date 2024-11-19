import re
from typing import List, Dict
import spacy
import time
import argparse
import logging
import os
from pydantic import BaseModel, Field
from tqdm import tqdm

from bench_generation.utils.tavily_search_models import TavilyClientWrapper
from bench_generation.utils.tools import load_or_generate, save_json_file, load_json_file

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

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
        sentences = re.split(r'[.!?]', text)
        sentence_count = len([s for s in sentences if s.strip()])
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        entity_count = len(set(entities))

        return (
            sentence_count >= min_sentences and 
            word_count >= min_words and 
            entity_count >= min_entities
        )

    def contains_sufficient_numerical_data(self, text: str, min_numbers: int) -> bool:
        # print ('----')
        # print (text)
        numbers = re.findall(
            r'\b(?!\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b)' 
            r'(?!\b\d{4}\b)'  
            r'\d+(?:,\d{3})*(?:\.\d+)?%?\b', text)
        distinct_numbers = set(numbers)
        return len(distinct_numbers) > min_numbers

    def filter_documents(self, documents: Dict[str, List[Dict[str, List[Dict[str, List[str]]]]]], 
                         min_numbers: int, min_sentences: int, min_words: int, min_entities: int) -> Dict[str, List[Dict[str, List[Dict[str, List[str]]]]]]:
        filtered_documents = {}
        total = 0
        count_dict = {}
        for main_topic, subtopics in documents.items():
            filtered_subtopics = []
            for subtopic_info in subtopics:
                subtopic = subtopic_info['subtopic']
                filtered_queries = []
                for query in subtopic_info['Queries']:
                    filtered_texts = []
                        # text for text in query['documents']
                        # if text and self.contains_sufficient_numerical_data(text, min_numbers) and
                        # self.contains_sufficient_context(text, min_sentences, min_words, min_entities)
                    # ]
                    for text in query['documents']:
                        # doc_ = ""
                        if text:
                            pieces = []
                            for piece in text:
                                if self.contains_sufficient_numerical_data(piece, min_numbers) and \
                                    self.contains_sufficient_context(piece, min_sentences, min_words, min_entities):
                                    pieces.append(piece)
                            if pieces!=[]:
                                doc_ = pieces[0]
                                filtered_texts.append(doc_)
                        

                    if filtered_texts:
                        filtered_queries.append({
                            "decomposable_query": query["decomposable_query"],
                            "atomic_queries": query["atomic_queries"],
                            "filtered_documents": filtered_texts[:] 
                        })
                        if len(filtered_texts) not in count_dict:
                            count_dict[len(filtered_texts)] = 1
                        else:
                            count_dict[len(filtered_texts)] += 1
                        total += len(filtered_texts)
                if filtered_queries:
                    filtered_subtopics.append({
                        "subtopic": subtopic,
                        "Queries": filtered_queries
                    })
            if filtered_subtopics:
                filtered_documents[main_topic] = filtered_subtopics
        logging.info(f"Number of filtered documents: {total}")
        logging.info(f"Stat: {count_dict}")
        return filtered_documents

    def fetch_page_content(self, query_dict: Dict[str, List[Dict[str, List[Dict[str, List[str]]]]]]) -> Dict[str, List[Dict[str, List[Dict[str, List[str]]]]]]:
        page_content_dict = {}
        search_count = 0

        for main_topic, subtopics in tqdm(query_dict.items(), desc="Processing fetch_page_content"):
            page_content_dict[main_topic] = []
            for subtopic_info in subtopics[:]:
                subtopic = subtopic_info['subtopic']
                queries = []
                for query in subtopic_info['Queries'][:]:
                    documents = []
                    for atomic_query in query['atomic_queries']:
                        result = self.tavily_client_wrapper.search(atomic_query)
                        documents.append(result if result else "")
                        if result:
                            search_count += 1
                        time.sleep(1)  
                    queries.append({
                        "decomposable_query": query["decomposable_query"],
                        "atomic_queries": query["atomic_queries"],
                        "documents": documents,
                    })
                page_content_dict[main_topic].append({
                    "subtopic": subtopic,
                    "Queries": queries
                })

        logging.info("Search completed.")
        logging.info(f"Total successful queries: {search_count}")
        return page_content_dict

if __name__ == "__main__":
    args = parse_args()

    query_dict_file = os.path.join('./outputs/data/', args.time_period, 'subtopics_and_queries.json')
    if os.path.exists(query_dict_file):
        query_dict = load_json_file(query_dict_file)
    else:
        logging.error(f"Query dictionary file not found: {query_dict_file}")
        query_dict = {}

    if not query_dict:
        logging.error("No queries found for document collection.")
        exit()
    
    tavily_client = TavilyClientWrapper()
    collector = DocumentCollector(tavily_client_wrapper=tavily_client)

    save_path = os.path.join(args.save_path, args.time_period, 'collected_documents.json')
    os.makedirs(args.save_path, exist_ok=True)

    document_data = load_or_generate(
        filename=save_path,
        condition=args.generate_documents_flag,
        generate_func=collector.fetch_page_content,
        query_dict=query_dict
    )

    logging.info(f"Searched documents saved to {save_path}")
    
    filtered_documents = collector.filter_documents(
        documents=document_data,
        min_numbers=args.min_numbers,
        min_sentences=args.min_sentences,
        min_words=args.min_words,
        min_entities=args.min_entities
    )
    
    filtered_documents_file = os.path.join(args.save_path, args.time_period, 'filtered_documents.json')
    save_json_file(filtered_documents_file, filtered_documents)
    logging.info(f"Filtered documents saved to {filtered_documents_file}")
