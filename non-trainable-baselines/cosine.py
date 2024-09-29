from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import pandas as pd
from fuzzywuzzy import fuzz

# Load the BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
import numpy as np
def get_bert_embedding(text):
    if pd.isna(text):
        return np.zeros((1, 768))

    # Tokenize and encode the text for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Get the output from the BERT model
    outputs = model(**inputs)
    # Use the mean of the last hidden states as the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def calculate_fuzzy_score(text1, text2):
    if pd.isna(text1) or pd.isna(text2):
        return 0  # Return 0 if either text is NaN
    return fuzz.ratio(text1, text2)
import sys 

dataset = sys.argv[1]
if(dataset == 'deli'):
    data_file = 'deli_dev_train.csv'
else:
    data_file = 'wtd_dev_train.csv'
data = pd.read_csv(data_file)
# Get embeddings for 'Probing_Question' and 'Causal_counterpart'
data['embedding_probing_question'] = data['Probing_Question'].apply(get_bert_embedding)
data['embedding_causal_counterpart'] = data['Causal_counterpart'].apply(get_bert_embedding)

# Calculate cosine similarity between embeddings
data['cosine_scores'] = [
    cosine_similarity(pq.reshape(1, -1), cc.reshape(1, -1))[0][0]
    for pq, cc in zip(data['embedding_probing_question'], data['embedding_causal_counterpart'])
]
data['fuzzy_token_overlap_score'] = data.apply(
    lambda row: calculate_fuzzy_score(row['Probing_Question'], row['Causal_counterpart']), axis=1)


# Display the updated dataframe with cosine similarity
print(data[['probingQuestionID', 'message_id', 'Probing_Question', 'Causal_counterpart', 'cosine_scores']].head())


dev_pairs = data[data['set'] == 'Dev']

average_cosine_score = dev_pairs['cosine_scores'].mean()
average_fuzzy_score = dev_pairs['fuzzy_token_overlap_score'].mean()
data.to_csv(f'{data_file}_withCosine.csv')
print(average_cosine_score, average_fuzzy_score)