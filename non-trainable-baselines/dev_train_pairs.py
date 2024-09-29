import pandas as pd
import sys
dataset = sys.argv[1]
# Load the datasets
if(dataset=='wtd'):
    final_path = '../wtd_dataset/final.csv'
    dataset_dev_train_pairs = '../wtd_Dev_train_pairs.csv'

else:
    final_path = '../deli_data/final.csv'
    dataset_dev_train_pairs = '../deli_Dev_train_pairs.csv'

dataset_dev_train_pairs = pd.read_csv(dataset_dev_train_pairs)
final_data = pd.read_csv(final_path)

# Display the first few rows of each dataset to understand their structure
dataset_dev_train_pairs.head(), final_data.head()


# Merge to add 'Probing_Question' based on 'probingQuestionID'
dataset_dev_train_pairs = dataset_dev_train_pairs.merge(final_data[['probingQuestionID', 'original_text']],
                                                on='probingQuestionID', how='left').rename(columns={'original_text': 'Probing_Question'})

# Merge to add 'Causal_counterpart' based on 'message_id'
dataset_dev_train_pairs = dataset_dev_train_pairs.merge(final_data[['message_id', 'original_text']],
                                                on='message_id', how='left').rename(columns={'original_text': 'Causal_counterpart'})

# Display the updated dataframe
dataset_dev_train_pairs.to_csv(f'{dataset}_dev_train.csv')
