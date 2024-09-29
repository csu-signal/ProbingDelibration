# Probing Deliberation Project

This repository contains the code and data for the project **“Any Other Thoughts, Hedgehog?” Linking Deliberation Chains in Collaborative Dialogues**, which focuses on identifying and modeling the causal relations between probing questions and their preceding causal utterances in collaborative task datasets. 

This README outlines the structure and key components of the repository.

## Repository Structure

```
ProbingDeliberation/
├── Data/
│   ├── deliDataset.csv
│   ├── wtDataset.csv
│
├── GPT-Annotation/
│   ├── GPT_probing_template.py
│   ├── WTD_GPT.py
│
├── non-trainable-baselines/
│   ├── cosine.py
│   ├── deli_dev_train.csv
│   ├── dev_train_pairs.py
│   ├── wtd_dev_train.csv_withCosine.csv
│
├── Trainable-Baselines/
│   ├── generate_gold_map.py
│   ├── generate_probing_csv.py
│   ├── generate_train_eval_pairs.py
│   ├── generateMessageID_WTD.py
│   ├── helper_probing.py
│   ├── probing_prediction.py
│   ├── probing_trainer.py
│   ├── probing_training_clustering.py
│
├── README.md
```
### Data Folder
- **deliDataset.csv**: Contains the *DeliData* corpus which focuses on group deliberation in problem-solving tasks.
- **wtDataset.csv**: Contains the *Weights Task Dataset* (WTD), a multimodal dataset where participants solve problems involving weighted blocks using a balance scale.

### GPT-Annotation Folder
- **GPT_probing_template.py**: A script for generating probing questions using GPT models, based on a template designed for task-specific dialogue processing.
- **WTD_GPT.py**: Annotates the WTD dataset by using GPT to identify probing questions and corresponding causal utterances.

### Non-Trainable Baselines Folder
- **cosine.py**: Implements a baseline model that calculates cosine similarity between utterances to identify potential causal relations.
- **deli_dev_train.csv**: Development data for training on the *DeliData* corpus.
- **dev_train_pairs.py**: Generates pairs of probing and causal utterances for the baseline models.
- **wtd_dev_train.csv_withCosine.csv**: Preprocessed WTD data that includes cosine similarity scores for utterance pairs.

### Trainable-Baselines Folder
- **generate_gold_map.py**: Generates a mapping between probing questions and their corresponding causal utterances for evaluation.
- **generate_probing_csv.py**: Creates a CSV file containing probing questions identified in the datasets.
- **generate_train_eval_pairs.py**: Generates training and evaluation pairs of utterances for use in the trainable baselines.
- **generateMessageID_WTD.py**: Maps message IDs for causal utterances in the *Weights Task Dataset*.
- **helper_probing.py**: Contains helper functions used in the probing task.
- **probing_prediction.py**: Predicts probing utterances from the dialogue data using a trainable model.
- **probing_trainer.py**: The main training script for the probing models, using the generated training data.
- **probing_training_clustering.py**: Implements the clustering mechanism to group probing questions with their causal counterparts.

### Requirements
Ensure you have the following installed before running the scripts:
- Python 3.8 or higher
- PyTorch (for training models)
- NumPy
- Pandas
- GPT-related libraries (e.g., `openai` for using GPT models)
- Cosine similarity libraries (e.g., `sklearn`)

### Running the Code

#### Baseline Models
1. **Cosine Similarity Baseline**: Run the script `cosine.py` to calculate cosine similarities between probing and causal utterances in the dataset. Example usage:
   ```bash
   python non-trainable-baselines/cosine.py
2. **Trainable Model**: Use probing_trainer.py to train models on the DeliData or Weights Task Dataset. Ensure that the training pairs are generated beforehand using generate_train_eval_pairs.py. 
Example usage:
```bash
   python Trainable-Baselines/probing_trainer.py