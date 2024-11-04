from pydantic import BaseModel, Field
from typing import List, Dict
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

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Document Summarization using LLMs.")
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of the model to use.')
    parser.add_argument('--llm_batch_size', type=int, default=5, help='Batch size for the LLM.')
    parser.add_argument('--save_path', type=str, default='./outputs/data/', help='Path to save data.')
    parser.add_argument('--generate_summary_flag', action='store_true', help='Generate summaries if set, otherwise load from file.')
    parser.add_argument('--filtered_documents_file', type=str, default='./outputs/data/March-2024-to-September-2024/filtered_documents.json', help='Path to the filtered documents file.')
    parser.add_argument('--output_file', type=str, default='./outputs/data/March-2024-to-September-2024/summarized_documents.csv', help='Path to save the summarized documents.')
    parser.add_argument('--document_file', type=str, default='./outputs/data/March-2024-to-September-2024/summarized_documents.csv', help='Path to save the summarized documents.')
    
    return parser.parse_args()

# Pydantic model for document summarization output
class DocumentSummary(BaseModel):
    summarized_content: str = Field(description="The polished or summarized content of the document. Don't include topic, subtopic, and query in the summarized_content.")

class DocumentSummarizer:
    @classmethod
    def get_prompt_template(cls) -> str:
        return """You are an AI tasked with either polishing or summarizing documents related to various topics and subtopics in less than 1000 words. The objective is to maintain the document's core content and ensure all numerical values are preserved, as these are crucial for generating mathematical reasoning problems.

Guidelines:
- For documents with fewer than 5000 words, focus on polishing: enhance clarity, coherence, and readability without significantly changing the word count or altering the original content.
- For documents exceeding 5000 words, summarize the content to a maximum of 5000 words. The summary must retain all key information and numerical values to preserve the document's original integrity.

For each document, consider the following:
- Retain all numerical data accurately, as this information is vital for downstream tasks.
- Ensure the polished or summarized content is relevant to the topic, subtopic, and query provided.
- Maintain the context and intent of the original document while improving the quality or conciseness of the text.

Input:
- **Topic:** {topic}
- **Subtopic:** {subtopic}
- **Document:** {document}

Example output format:
```json
{{
  "summarized_content": ""
}}
```

{format_instructions}
"""

    @classmethod
    def summarize_documents(cls, llm, filtered_documents: Dict[str, Dict[str, Dict[str, List[str]]]]) -> Dict:
        """Summarizes filtered documents using the LLM."""
        parser = PydanticOutputParser(pydantic_object=DocumentSummary)
        prompt_template = PromptTemplate(
            template=cls.get_prompt_template(),
            input_variables=["topic", "subtopic", "query", "document", "format_instructions"]
        )
        
        # Convert nested dictionary to DataFrame for easier processing
        data = []
        error_c = 0
        document_data = []
        document_count = 0
        for topic, subtopics in filtered_documents.items():
            for subtopic_info in subtopics:
                # logging.info(f"subtopic_info: {subtopic_info}")  
                subtopic = subtopic_info['subtopic']
                for query in subtopic_info['Queries']:
                    doc_ids = []
                    summarized_documents = [] 
                    for document in query['filtered_documents']:
                        input_x = prompt_template.format(
                            topic=topic, 
                            subtopic=subtopic, 
                            document=document,
                            format_instructions=parser.get_format_instructions()
                        )
                        message_list = [{"role": "user", "content": input_x}]
                        response = llm.call_llm_api(message_list, temperature=0, max_tokens=4096)

                        response_json = extract_json_from_string(response)
                        if response_json is not None:
                            response_str = response_json.get("summarized_content", "")
                            print ("# ok")
                        else:
                            response_str = ""
                            error_c+=1
                            print ("# errors:", error_c)
                        summarized_documents.append(response_str)

                        doc_id = 'Doc_'+str(document_count)
                        doc_ids.append(doc_id)
                        document_count+=1


                    data.append({
                        'Topic': topic,
                        'Subtopic': subtopic,
                        'decomposable_query': query['decomposable_query'],
                        'atomic_queries': query['atomic_queries'],
                        'filtered_documents': query['filtered_documents'],
                        'summarized_documents': summarized_documents,
                        'doc_ids': doc_ids,
                    })
        
        
        
        return data

    @classmethod
    def document_df_construction(cls, data) -> pd.DataFrame:
        document_data = []
        for idx, elem in enumerate(data):
            filtered_documents = elem["filtered_documents"]
            summarized_documents = elem["summarized_documents"]
            doc_ids = elem["doc_ids"]
            for ii in range(len(filtered_documents)):
                doc_id = doc_ids[ii]
                document = filtered_documents[ii]
                summarized_document = summarized_documents[ii]
                document_data.append({
                    'Document_ID': doc_id,
                    'Document': document,
                    'Summarized_document': summarized_document, 
                    })
        # document_data_df = pd.DataFrame(document_data[:])
        return document_data


if __name__ == "__main__":
    args = parse_args()
    
    # Load filtered documents from file
    logging.info(f"Summarizing documents starts")
    if os.path.exists(args.filtered_documents_file):
        filtered_documents = load_json_file(args.filtered_documents_file)
    else:
        logging.error(f"Filtered documents file not found: {args.filtered_documents_file}")
        filtered_documents = {}

    if not filtered_documents:
        logging.error("No filtered documents found for summarization.")
        exit()

    # Initialize the OpenAI client wrapper with config
    config = {
        "model_name": args.model_name,
        "llm_batch_size": args.llm_batch_size,
    }
    
    llm = OpenAIClientWrapper(config)
    
    # Load or generate document summaries
    data = load_or_generate(
        filename=args.output_file,
        condition=args.generate_summary_flag,
        generate_func=DocumentSummarizer.summarize_documents,
        llm=llm,
        filtered_documents=filtered_documents
    )

    document_data = DocumentSummarizer.document_df_construction(data)

    # summarized_df['Document_ID'] = 'DOC_' + summarized_df.index.astype(str)
    # Save summarized documents to CSV
    # document_data_df.to_csv(args.document_file, index=False)
    save_json_file(args.document_file, document_data)
    logging.info(f"Summarized documents saved to {args.output_file}")
