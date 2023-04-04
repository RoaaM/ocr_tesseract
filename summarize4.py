import csv
import nltk
from nltk.corpus import stopwords
import re
from collections import defaultdict
import networkx as nx
from transformers import pipeline, set_seed

nltk.download('popular')

def hypergraph_abstractive_summarization(text, summary_ratio):
    # Tokenize the text into sentences and words using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Remove stopwords and punctuation from the sentences
    stop_words = set(stopwords.words('english'))
    sentences = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences if len(sentence) > 0]
    sentences = [sentence for sentence in sentences if len(sentence) > 0 and not all(word in stop_words for word in sentence.split())]

    # Apply the BART abstractive summarization model to the sentences
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    summary = summarizer(' '.join(sentences), max_length=1024, min_length=1, do_sample=False, num_beams=4, length_penalty=2.0, early_stopping=True)[0]['summary_text']

    return summary

# Set the summary ratio
summary_ratio = 0.20

# Open the input CSV file
with open('output.csv', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)

    # Create a new CSV file for the output
    with open('output_summary4.csv', 'w', newline='', encoding='utf-8-sig') as outfile:
        fieldnames = ['Text', 'Summary']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each row in the input CSV file
        for row in reader:
            # Get the text from the input CSV file
            text = row['Text']

            # Generate the summary using the hypergraph-based abstractive summarization algorithm
            summary = hypergraph_abstractive_summarization(text, summary_ratio)

            # Write the original text and summary to the output CSV file
            writer.writerow({'Text': text, 'Summary': summary})
