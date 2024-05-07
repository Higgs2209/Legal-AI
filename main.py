from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from datasets import load_dataset
import json
from langchain_community.llms import HuggingFacePipeline
import concurrent.futures
from tqdm import tqdm
from transformers import pipeline

# Maximum number of tokens in a chunk
max_tokens = 3000
#tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
#splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)


# Load your dataset
dataset = load_dataset('umarbutler/open-australian-legal-corpus')

dataset = dataset['train'].select(range(2))

# Select the specific column you want to process
#column_name = 'text'
#document_text = dataset[column_name]
#document_citation = dataset['citation']
#document_jurisdiction = dataset['jurisdiction']


#text_chunks = splitter.chunks(document_text)

# print(text_chunks)

#Define the prompt template
prompt_template = """
# Snippet
The snippet from an Australian legal document from which you must synthesise a question and answer is provided below.
<document_metadata>
<document_title>{citation}</document_title>
<document_jurisdiction>{jurisdiction}</document_jurisdiction>
<document_type>{type}</document_type>
</document_metadata>
<snippet>
{text}
</snippet>

# Format
You must format your response as follows:
<format>
# Question
{{A question related to the snippet, or a topic discussed therein.}}

# Answer
{{The answer to the question, extracted from the snippet.}}
</format>

# Instructions
You must act as a question-and-answer synthesiser that takes a snippet from an Australian legal document and synthesises a question related to the snippet, or a topic discussed therein, and an answer to that question, extracted from the snippet.

Your question must be decontextualised and standalone from the snippet. If the question pertains to a particular jurisdiction or document, it must state that explicitly (eg, 'In Victoria, is it lawful for ...?', 'What did the Court decide in Mabo v Queensland (No 2) [1992] HCA 23?', etc...).

Your answer must also be decontextualised and standalone from the snippet. It must reference the document from which it came (eg, 'Under the Crimes Act 1958 (Vic), ...', 'In Mabo v Queensland (No 2) [1992] HCA 23, the Court decided ...', etc...), not the snippet itself. It must be capable of being understood on its own and without reference to the snippet or its source document.

When referring to a document (eg, the Crimes Act) or a part thereof (eg, Paragraph 1), or to a person (eg, the Minister), organisation (eg, the Department) or concept (eg, the rule of law), you must refer to it by its full name (eg, the Crimes Act 1958 (Vic) instead of the Crimes Act, Paragraph 1 of ABC v XYZ instead of Paragraph 1, the Commonwealth Minister for Finance instead of the Minister).

If it is not possible to synthesise a question and answer from the snippet, you must respond with `<!no_qa!>`. Otherwise, your response must conform to the provided format.
"""



import os
import torch
from concurrent.futures import ThreadPoolExecutor
from datasets import concatenate_datasets

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def process_row(i):
    responses = []

    # Initialize the pipeline and splitter inside the function
    model_pipeline = pipeline("text-generation", model="cognitivecomputations/dolphin-2.2.1-mistral-7b", tokenizer="cognitivecomputations/dolphin-2.2.1-mistral-7b", token="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", max_length=5000)
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

    # Get the data from the current row
    citation = dataset['citation'][i]
    jurisdiction = dataset['jurisdiction'][i]
    type = dataset['type'][i]  # The type is provided in the dataset

    # Get the text and split it into chunks
    text = dataset['text'][i]
    text_chunks = splitter.chunks(text)

    # Iterate over each chunk
    for chunk in text_chunks:
        # Format the prompt with the current row's data and the current chunk
        prompt = prompt_template.format(citation=citation, jurisdiction=jurisdiction, type=type, text=chunk)
        print (prompt)
        # Run the prompt through the pipeline
        response = model_pipeline(prompt)

        responses.append(response)
        print(response)
    return {"responses": responses}

# Create a ThreadPoolExecutor with a limited number of workers
# Create a ThreadPoolExecutor with a limited number of workers
with ThreadPoolExecutor(max_workers=4) as executor:
    # Use a tqdm progress bar
    for i in tqdm(range(len(dataset))):
        # Process the current row and get the responses
        result = executor.submit(process_row, i).result()
        responses = result['responses']  # Extract the list of responses from the result dictionary
        print(responses)

        # Open the .jsonl file and write the responses
        with open('responses.jsonl', 'a') as f:
            for response in responses:
                f.write(json.dumps(response) + '\n')