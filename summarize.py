import csv
import nltk
from nltk.corpus import stopwords
import re
from collections import defaultdict
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
                    hypergraph[i].append((j, weight))

    # Apply the PageRank algorithm to the hypergraph
    pageranks = defaultdict(float)
    for i in range(len(sentences)):
        pageranks[i] = 1.0 / len(sentences)

    damping_factor = 0.85
    epsilon = 1e-5
    num_iterations = 100
    for _ in range(num_iterations):
        new_pageranks = defaultdict(float)
        for i in range(len(sentences)):
            incoming_weights = sum(weight for _, weight in hypergraph[i])
            for j, weight in hypergraph[i]:
                new_pageranks[j] += damping_factor * pageranks[i] * weight / incoming_weights
        max_diff = max(abs(pageranks[i] - new_pageranks[i]) for i in range(len(sentences)))
        if max_diff <= epsilon:
            break
        pageranks = new_pageranks

    # Sort the sentences by PageRank scores and generate the summary
    ranked_indices = sorted(range(len(sentences)), key=lambda i: pageranks[i], reverse=True)
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
    with open('output_summary.csv', 'w', newline='', encoding='utf-8-sig') as outfile:
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
