from rouge import Rouge
import pandas as pd


# Initialize Rouge
rouge = Rouge()

# Get the original text and summary from the output CSV file
df = pd.read_csv('output_summary2.csv', encoding='utf-8-sig')
texts = df['Text'].tolist()
summaries = df['Summary'].tolist()

# Calculate the Rouge scores for the summary and the original text
scores = rouge.get_scores(summaries, texts)

# Initialize the counters for the metrics
rouge1_f1_sum = 0
rouge1_p_sum = 0
rouge1_r_sum = 0
rouge2_f1_sum = 0
rouge2_p_sum = 0
rouge2_r_sum = 0
rougeL_f1_sum = 0
rougeL_p_sum = 0
rougeL_r_sum = 0

# Iterate over the scores for each summary and update the counters
for score in scores:
    rouge1_f1_sum += score['rouge-1']['f']
    rouge1_p_sum += score['rouge-1']['p']
    rouge1_r_sum += score['rouge-1']['r']
    rouge2_f1_sum += score['rouge-2']['f']
    rouge2_p_sum += score['rouge-2']['p']
    rouge2_r_sum += score['rouge-2']['r']
    rougeL_f1_sum += score['rouge-l']['f']
    rougeL_p_sum += score['rouge-l']['p']
    rougeL_r_sum += score['rouge-l']['r']

# Calculate the average of the metrics
num_summaries = len(summaries)
rouge1_f1_avg = rouge1_f1_sum / num_summaries
rouge1_p_avg = rouge1_p_sum / num_summaries
rouge1_r_avg = rouge1_r_sum / num_summaries
rouge2_f1_avg = rouge2_f1_sum / num_summaries
rouge2_p_avg = rouge2_p_sum / num_summaries
rouge2_r_avg = rouge2_r_sum / num_summaries
rougeL_f1_avg = rougeL_f1_sum / num_summaries
rougeL_p_avg = rougeL_p_sum / num_summaries
rougeL_r_avg = rougeL_r_sum / num_summaries

# Print the average metrics
print(f"Average Rouge-1 precision: {rouge1_p_avg:.4f}")
print(f"Average Rouge-1 recall: {rouge1_r_avg:.4f}")
print(f"Average Rouge-1 F1 score: {rouge1_f1_avg:.4f}")
print(f"Average Rouge-2 precision: {rouge2_p_avg:.4f}")
print(f"Average Rouge-2 recall: {rouge2_r_avg:.4f}")
print(f"Average Rouge-2 F1 score: {rouge2_f1_avg:.4f}")
print(f"Average Rouge-L precision: {rougeL_p_avg:.4f}")
print(f"Average Rouge-L recall: {rougeL_r_avg:.4f}")
print(f"Average Rouge-L F1 score: {rougeL_f1_avg:.4f}")
