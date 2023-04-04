import csv
import nltk
from nltk.corpus import stopwords
import re
from collections import defaultdict
import numpy as np

nltk.download('popular')

def textrank_hypergraph_summarization(text, summary_ratio):
    # Tokenize the text into sentences and words using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove stopwords and punctuation from the words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]

    # Build the word co-occurrence graph
    co_occurrences = defaultdict(int)
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if i != j:
                co_occurrences[(words[i], words[j])] += 1

    # Build the sentence co-occurrence graph
    sentence_indices = defaultdict(list)
    for i, sentence in enumerate(sentences):
        for word in words:
            if word in sentence.lower():
                sentence_indices[word].append(i)

    # Build the hypergraph
    hypergraph = defaultdict(list)
    for (w1, w2), weight in co_occurrences.items():
        for i in sentence_indices[w1]:
            for j in sentence_indices[w2]:
                if i != j:
                    hypergraph[i].append(j)

    # Calculate the degree of each sentence in the hypergraph
    sentence_degrees = defaultdict(int)
    for i in range(len(sentences)):
        sentence_degrees[i] = len(hypergraph[i])

    # Calculate the similarity matrix
    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                intersection = set(hypergraph[i]).intersection(set(hypergraph[j]))
                union = set(hypergraph[i]).union(set(hypergraph[j]))
                if len(union) != 0:
                    jaccard = len(intersection) / len(union)
                    sim_matrix[i][j] = jaccard
                    sim_matrix[j][i] = jaccard

    # Apply the LexRank algorithm to the similarity matrix
    alpha = 0.85
    epsilon = 1e-5
    scores = np.ones(len(sentences))
    old_scores = np.zeros(len(sentences))
    while np.sum(np.abs(scores - old_scores)) > epsilon:
        old_scores = np.copy(scores)
        for i in range(len(sentences)):
            score = 0.0
            for j in range(len(sentences)):
                if sim_matrix[i][j] != 0:
                    score += alpha * sim_matrix[i][j] * old_scores[j] / sentence_degrees[j]
            score += 1 - alpha
            scores[i] = score

    # Sort the sentences by scores and generate the summary
    ranked_indices = np.argsort(scores)[::-1]
    num_sentences = max(1, int(summary_ratio * len(sentences)))
    selected_indices = sorted(ranked_indices[:num_sentences])
    summary = ' '.join(sentences[i] for i in selected_indices)

    return summary


# Set the summary ratio
summary_ratio = 0.2

# Open the input CSV file
with open('output.csv', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)

    # Create a new CSV file for the output
    with open('output_summary3.csv', 'w', newline='', encoding='utf-8-sig') as outfile:
        fieldnames = ['Text', 'Summary']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each row in the input CSV file
        for row in reader:
            # Get the text from the input CSV file
            text = row['Text']

            # Generate the summary using the hypergraph-based TextRank algorithm
            summary = textrank_hypergraph_summarization(text, summary_ratio)

            # Write the original text and summary to the output CSV file
            writer.writerow({'Text': text, 'Summary': summary})
