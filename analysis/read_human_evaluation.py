import pandas as pd

# Path to the Excel file
xlsx_file_path = "./analysis/annotated_files/human_evaluation_annotations_sssd(annotated).xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(xlsx_file_path, engine='openpyxl')

# Display the first few rows
print(df.iloc[0])
