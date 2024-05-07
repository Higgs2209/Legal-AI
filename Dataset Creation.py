import datasets
from datasets import load_dataset
from transformers import BertTokenizerFast
from concurrent.futures import ThreadPoolExecutor
import json

# Step 1: Load the existing dataset
dataset = load_dataset('umarbutler/open-australian-legal-corpus')

# Step 2: Define a function to split the text into chunks of 512 tokens
tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-large-cased')


def chunk_text(text):
    # Account for special tokens
    max_length = 512 - tokenizer.num_special_tokens_to_add(pair=False)
    tokens = tokenizer.tokenize(text)
    return [tokenizer.convert_tokens_to_string(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]


from concurrent.futures import ThreadPoolExecutor

# Define a global executor
executor = ThreadPoolExecutor(max_workers=6)

def process_row(row):
    # Submit the task to the executor and return the future
    future = executor.submit(chunk_text, row['text'])
    # Wait for the task to complete and get the result
    chunks = future.result()
    return {"text_chunks": chunks}

# Apply the function to the dataset
new_dataset = dataset.map(process_row)


with ThreadPoolExecutor(max_workers=6) as executor:
    new_dataset = dataset.map(executor.submit(process_row))

# Step 4: Save the new dataset
new_dataset.save_to_disk('/')

# Step 5: Upload the dataset to Hugging Face

dataset_dict = datasets.DatasetDict({"split": new_dataset})

dataset_dict.save_to_disk("/")
dataset_dict.upload_to_hub("Higgs201/tokenised-open-australian-legal-corpus", "This is a test dataset")
