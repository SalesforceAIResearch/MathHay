import yaml
import json
import re
import os
import pandas as pd
from itertools import combinations
import random

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)    

def load_jsonl_file(save_path):
    # save_path = f"./results/{task}_haystack.jsonl"
    with open(save_path, 'r') as file:
        data_rows = [json.loads(line) for line in file]  # Load the haystack data rows
    return data_rows

def save_jsonl_file(file_path, data_rows):
    with open(file_path, 'w') as file:
        for row in data_rows:
            file.write(json.dumps(row) + '\n')  # Save each row as a separate JSON object in the JSONL file


def extract_json_from_string(text):
    # Use regex to find JSON-like patterns (adjusted for better capturing)
    json_patterns = re.findall(r'```(?:js|json)?\n(.*?)```|({.*})', text, re.DOTALL)
    
    extracted_jsons = []
    for match in json_patterns:
        # Join non-empty parts of the match
        json_str = ''.join([part for part in match if part])
        
        try:
            # Remove invalid control characters (like newlines inside strings)
            json_str = re.sub(r'[\n\r]', '', json_str)
            # Parse JSON string to Python dictionary
            json_obj = json.loads(json_str)
            extracted_jsons.append(json_obj)
        except json.JSONDecodeError:
            print("Invalid JSON found, skipping:", json_str)
    
    # Return the first valid JSON found, if any
    return extracted_jsons[0] if extracted_jsons else None

# def load_or_generate(filename, condition, generate_func, *args, **kwargs):
#     """
#     Load content from a file if it exists and the condition is not met; otherwise, generate and save it.
    
#     Parameters:
#     - filename (str): The path to the file for saving/loading.
#     - condition (bool): Whether to generate new content (if True) or load from the file (if False).
#     - generate_func (function): The function used to generate the content.
#     - *args, **kwargs: Arguments and keyword arguments for the generate function.
    
#     Returns:
#     - The generated or loaded content.
#     """
    
#     file_path = filename #os.path.join(results_folder, filename)

#     if os.path.exists(file_path) and not condition:
#         with open(file_path, 'r') as file:
#             return json.load(file)
#     else:
#         content = generate_func(*args, **kwargs)
#         with open(file_path, 'w') as file:
#             json.dump(content, file, indent=2)
#         return content

def load_or_generate(filename, condition, generate_func, *args, **kwargs):
    """
    Load content from a file if it exists and the condition is not met; otherwise, generate and save it.
    
    Parameters:
    - filename (str): The path to the file for saving/loading.
    - condition (bool): Whether to generate new content (if True) or load from the file (if False).
    - generate_func (function): The function used to generate the content.
    - *args, **kwargs: Arguments and keyword arguments for the generate function.
    
    Returns:
    - The generated or loaded content.
    """
    
    file_path = filename

    if os.path.exists(file_path) and not condition:
        if filename.endswith('.csv'):
            return pd.read_csv(file_path)
        elif filename.endswith('.json'):
            with open(file_path, 'r') as file:
                return json.load(file)
        else:
            raise ValueError("Unsupported file format. Please use .csv for DataFrame or .json for dict.")
    else:
        content = generate_func(*args, **kwargs)
        if isinstance(content, pd.DataFrame):
            # Save as CSV if the content is a DataFrame
            content.to_csv(file_path, index=False)
        elif isinstance(content, (dict, list)): 
            # Save as JSON if the content is a dictionary
            with open(file_path, 'w') as file:
                json.dump(content, file, indent=2)
        else:
            raise ValueError("Generated content must be a DataFrame or a dictionary.")
        
        return content


def combine_rows(df, combination_size=2, sample_size=5):
    """
    Combines rows under the same subtopic into random groups of a specified size.

    Parameters:
    df (pd.DataFrame): The input DataFrame with the columns 
                       ['Document_ID', 'Topic', 'Subtopic', 'Query', 'Document', 
                        'TokenLen_Document', 'Summarized_Document'].
    combination_size (int): The number of rows to combine (2 or 3).
    sample_size (int): The number of random combinations to take (default is 5).

    Returns:
    pd.DataFrame: A new DataFrame with combined rows.
    """
    # Check if combination size is valid
    if combination_size not in [2, 3]:
        raise ValueError("Combination size must be 2 or 3.")

    combined_rows = []

    # Group by 'Topic' and 'Subtopic'
    for (topic, subtopic), group in df.groupby(['Topic', 'Subtopic']):
        # if topic not in ['Financial Market Analysis']:
        #     continue
        # if subtopic not in ["Trends in Stock Prices"]:
        #     continue
        # Generate all combinations of the specified size within each group
        all_combinations = list(combinations(group.iterrows(), combination_size))


        
        # Sample up to `sample_size` combinations randomly
        random.seed(2024)
        sampled_combinations = random.sample(all_combinations, min(sample_size, len(all_combinations)))

        for combination in sampled_combinations:
            # Create a dictionary for the combined row
            combined_row = {
                'Topic': topic,
                'Subtopic': subtopic,
            }

            # Add details for each document in the combination
            for i, (index, row) in enumerate(combination, start=1):
                combined_row[f'Document_ID{i}'] = row['Document_ID']
                combined_row[f'Document{i}'] = row['Document']
                combined_row[f'Summarized_Document{i}'] = row['Summarized_Document']
                combined_row[f'Query{i}'] = row['Query']
            
            combined_rows.append(combined_row)

    # Convert the list of combined rows into a new DataFrame
    combined_df = pd.DataFrame(combined_rows)
    
    return combined_df