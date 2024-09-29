import pickle
import torch
from helper_probing import tokenize, tokenize_utterances, forward_ab, f1_score, \
accuracy, precision, recall, cluster, generate_key_file, get_bert_embedding, get_cosine_similarity, \
calculate_fuzzy_score, calculate_iou
from generate_gold_map import generate_gold_map
from probing_prediction import predict_causal_counterpart
import random
from tqdm import tqdm
import os
from models import CrossEncoder
from delitoolkit.delidata import DeliData
import pickle
import json 
import pandas as pd
import numpy as np
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
from sklearn.metrics.pairwise import cosine_similarity
import sys

from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import pandas as pd
from fuzzywuzzy import fuzz

# Load the BERT model and tokenizer
cosine_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(cosine_model_name)
cosine_model = BertModel.from_pretrained(cosine_model_name)
import numpy as np

from collections import defaultdict




def generate_pairs_for_train_eval(gold_map, split):
    '''
    Generates all antecedent pairs and binary labels of casual and probing interventions using the gold labels
    generated with "get_gold_map_cldeaned".
    Outputs pairs, labels and causal and probing label for each train/eval pair
     
    '''
    group_to_interventions = defaultdict(list)
    pairs_labels_dict = {}
    pair_sample = []
    split_intervention_ids = sorted([m_id for m_id, m in gold_map.items() if m['split'] == split])

    for intervention_id in split_intervention_ids:
        group_id = gold_map[intervention_id]['group_id']
        group_to_interventions[group_id].append(intervention_id)
    intervention_pairs = []    
    intervention_labels = []
    causal_probing_label = { }
    probing_label = {}
    intervention_pairs_labels = {}
    c =0
    #group_to_interventions = list(group_to_interventions.items())[0:10]
    #group_to_interventions = dict(group_to_interventions)
    for interventions in group_to_interventions.values() :
        list_interventions = list(interventions)
        for i in range(len(list_interventions)):
            for j in range(i + 1):
                if i != j:
                    intervention_pairs.append((list_interventions[i], list_interventions[j])) # add the pair label and the causal label 

                    intervention_labels.append(int(gold_map[list_interventions[i]]['gold_cluster'] ==gold_map[list_interventions[j]]['gold_cluster']) )

    for intervention in split_intervention_ids:
        if len(intervention.split("_")) > 1:
            causal_probing_label[intervention] = 'probing'
        else:
            causal_probing_label[intervention] = 'causal'
    print(f" {split} final pairs", len(intervention_pairs), len(intervention_labels), len(causal_probing_label))
    return intervention_pairs, intervention_labels, causal_probing_label

def read(key, response):
    return get_coref_infos('%s' % key, '%s' % response,
            False, False, True)

def get_gold_map(test_dict, probing_map, document_map):
    
    m_id_occurrences = defaultdict(list)
    gold_cluster_map = {}
    m_id_to_gold_cluster_map = {}
    final_gold_map_new = {}
    # Iterate through the gold_cluster_map dictionary

    for x, info in test_dict.items():
        m_id_occurrences[x[1]].append(x[0])

    # Find primary keys (x) where the 'm_id' value occurs more than once
    duplicate_x_keys = [x_values for m_id, x_values in m_id_occurrences.items() if len(x_values) > 1]

    for  m_id, x_values in m_id_occurrences.items():

        if len(x_values) > 1:
            duplicate_x_keys = x_values
            min_elements = []
            min_element = None
            min_digit = float('inf') 

            # Iterate through each sublist in the data list
            for item in duplicate_x_keys:
                # Split the element by underscores
                parts = item.split('_')
                # Extract the digit after the underscore
                digit = int(parts[-1])
                # Check if the digit is smaller than the current minimum
                if digit < min_digit:
                    #print("min digit", min_digit, m_id, item)
                    min_digit = digit
                    #print("min digit assign", min_digit, m_id, item)
                    min_element = item

                # Append the minimum element to the result list
                    min_elements.append(min_element)
                    #print(min_element)
                    m_id_to_gold_cluster_map[m_id] = min_element
          
    for index, (x,y) in enumerate(test_dict.items()):
        #print(index, x, y)
        final_gold_map_new[x[0]] = {'m_id':x[1], 'gold_cluster': x[0], 'group_id': document_map[x[1]]['group_id']}
        if x[1] in m_id_to_gold_cluster_map.keys():
            #print(y, x)

            #print("this loop is getting", m_id_to_gold_cluster_map[x[1]] ,x[1], x[0])
            final_gold_map_new[x[1]] = {'m_id':x[1], 'gold_cluster': m_id_to_gold_cluster_map[x[1]], 'group_id': document_map[x[1]]['group_id']}
        else:
            #print(x[0], x[1], "else loop")

            final_gold_map_new[x[1]] = {'m_id':x[1], 'gold_cluster': x[0], 'group_id': document_map[x[1]]['group_id']}  
    return final_gold_map_new



def resample_split(test_dict, gold_map): 

    total_correct_gold = []
    total_correct_negative = []
    new_test_pairs_dict = {}
    for x, y in test_dict.items():
        if y==1:

            #get gold for x and y 
            gold_probing = gold_map[x[0]]['gold_cluster']
            gold_cc = gold_map[x[1]]['gold_cluster']
            #print(x[0],x[1], "gold ***", gold_probing, gold_cc, y)
            if gold_probing == gold_cc:
                #print(x[0],x[1], "gold coorect paur ***", gold_probing, gold_cc, y)
                total_correct_gold.append(1)
                new_test_pairs_dict[x] = y
            else:
                new_test_pairs_dict[x] = 0
        else:
            gold_probing = gold_map[x[0]]['gold_cluster']
            gold_cc = gold_map[x[1]]['gold_cluster']
            if gold_probing != gold_cc:
                total_correct_negative.append(1)
                new_test_pairs_dict[x] = y
            else:
                new_test_pairs_dict[x] = 1
    return new_test_pairs_dict
            

def generate_split_pairs_labels(dataset, causal_counterpart_map, document_map, gold = False):

    train_pairs = []
    train_labels = []
    dev_pairs = []
    dev_labels  = []
    test_pairs = []
    test_labels = []
    pos_pairs = pd.read_csv(f"{dataset}/positive_samples.csv")
    neg_pairs = pd.read_csv(f"{dataset}/negative_samples.csv")
    pos_pairs['probingQuestionID'].tolist()
    # pairs = zip([(pos_pairs['probingQuestionID'].tolist(), pos_pairs['message_id'].tolist())])
    positive_pairs = list(zip(pos_pairs['probingQuestionID'], pos_pairs['message_id']))
    negative_pairs = list(zip(neg_pairs['probingQuestionID'], neg_pairs['message_id']))
    
    
    
    # Filtered training  without the specified indices
    
    def remove_initial_message_from_pairs(positive_pairs,negative_pairs):
    
        pos_training_samples = []
        neg_training_samples = []
        exceptions_pairs = []
        negative_pairs_exception = []
        pos_bad_indices = []
        neg_bad_indices = []

        for index, (x, y) in enumerate(positive_pairs):
            if y in document_map.keys():
                continue
            elif  y == '-1':
                pos_bad_indices.append(index)
            else:
                pos_bad_indices.append(index)
        for index, (x, y) in enumerate(negative_pairs):
            if y in document_map.keys():
                continue
            else:
                neg_bad_indices.append(index)
        return pos_bad_indices, neg_bad_indices
    pos_bad_indices, neg_bad_indices = remove_initial_message_from_pairs(positive_pairs,negative_pairs)
    #print("bad indices", len(pos_bad_indices), len(neg_bad_indices))
    #print(pos_bad_indices)
   
    positive_pairs_cleaned = [(x,y) for index, (x,y) in enumerate(positive_pairs) if index not in pos_bad_indices]
    negative_pairs_cleaned = [(x,y) for index, (x,y) in enumerate(negative_pairs) if index not in neg_bad_indices]
    #print(len(positive_pairs_cleaned), len(negative_pairs_cleaned))
    #print(positive_pairs_cleaned)
    if not gold: 
        for index, (x, y) in enumerate(positive_pairs_cleaned):
            if causal_counterpart_map[x]['set'] =="Train":
                train_pairs.append((x,y))
                train_labels.append(1)
            elif causal_counterpart_map[x]['set'] =="Dev":
                dev_pairs.append((x,y))
                dev_labels.append(1)

            elif causal_counterpart_map[x]['set'] =="Test":
                test_pairs.append((x,y))
                test_labels.append(1)


        for index, (x, y) in enumerate(negative_pairs_cleaned): # get the negative pairwise labels as zero 
            if causal_counterpart_map[x]['set'] =="Train":
                train_pairs.append((x,y))
                train_labels.append(0)
            elif causal_counterpart_map[x]['set'] =="Dev":
                dev_pairs.append((x,y))
                dev_labels.append(0)

            elif causal_counterpart_map[x]['set'] =="Test":
                test_pairs.append((x,y))
                test_labels.append(0)
    else:
        for index, (x, y) in enumerate(positive_pairs_cleaned):
            if causal_counterpart_map[x]['set'] =="Train":
                train_pairs.append((x,y))
                train_labels.append(1)
            elif causal_counterpart_map[x]['set'] =="Dev":
                dev_pairs.append((x,y))
                dev_labels.append(1)

            elif causal_counterpart_map[x]['set'] =="Test":
                test_pairs.append((x,y))
                test_labels.append(1)


        for index, (x, y) in enumerate(negative_pairs_cleaned): # get the negative pairwise labels as zero 
            if causal_counterpart_map[x]['set'] =="Train":
                train_pairs.append((x,y))
                train_labels.append(0)
    return train_pairs, train_labels, dev_pairs, dev_labels, test_pairs, test_labels


def combine_non_probing_with_probing_map(dataset, wtd = False):
    #print('DOING FOR WTD ', wtd)
    causal_counterpart_gpt_responses_file = f'{dataset}/final.pkl'
    with open(causal_counterpart_gpt_responses_file, "rb") as f:
        generated_causal_counterpart_map  = pickle.load(f)

  
    utterance_map_train = {m_id: m for m_id, m in generated_causal_counterpart_map.items() if m['set'] == 'Train'}
    utterance_map_dev = {m_id: m for m_id, m in generated_causal_counterpart_map.items() if m['set'] == 'Dev'}
    utterance_map_test = {m_id: m for m_id, m in generated_causal_counterpart_map.items() if m['set'] == 'Test'}

    group_id_train = set([y['group_id'] for x, y in utterance_map_train.items()])
    group_id_dev = set([y['group_id'] for x, y in utterance_map_dev.items()])
    group_id_test = set([y['group_id'] for x, y in utterance_map_test.items()])

    delidata_corpus = DeliData()
    groups = list(delidata_corpus.corpus.keys())
    probing_questions = []
    document_map = {}
    wtd = False
    for group, messages in delidata_corpus.corpus.items():
        for m in messages:
#             if m['annotation_type'] != 'Probing':
            if m['message_id'] != '-1':
                document_map[m['message_id']] = {
                                'group_id': m['group_id'],
                                'message_id': m['message_id'],
                                'message_type': m['message_type'],
                                'origin': m['origin'],
                                'original_text': m['original_text'],
                                'clean_text': m['clean_text'],
                                'annotation_type': m['annotation_type'],
                                'annotation_target': m['annotation_target'],
                                'annotation_additional': m['annotation_additional'],
                                'team_performance': m['team_performance'],
                                'performance_change': m['performance_change'],
                                'sol_tracker_message': m['sol_tracker_message'],
                                'sol_tracker_all': m['sol_tracker_all']


        #                         'prev_uttterance_history': prev_history_map[key]
                            }
                
    if(wtd):
        df = pd.read_csv(f'{dataset}/final.csv')

        for _, row in df.iterrows():
            document_map[row['message_id']] = {
                'group_id': row['group_id'],
                'message_id': row['message_id'],
               # 'message_type': row['message_type'],
                'origin': row['origin'],
                'original_text': row['original_text'],
                #'clean_text': row['clean_text'],
                'annotation_type': row['annotation_type'],
                # 'prev_utterance_history': prev_history_map[key]  # Uncomment and define prev_history_map if needed
            }

    for m_id, values in document_map.items():
        if values['group_id'] in group_id_train:
            document_map[m_id]['set'] ='Train'
        elif values['group_id'] in group_id_dev:
            document_map[m_id]['set'] ='Dev'
        elif values['group_id'] in group_id_test:
            document_map[m_id]['set'] ='Test' 
    return generated_causal_counterpart_map, document_map


def get_probing_causal_counterpart_clusters(split,test_pairs, test_scores_ab, test_scores_ba, gold_map_test, working_folder, threshold = .5):

    split = 'test'
    test_score_map = {}
    for b, ab, ba in zip(test_pairs, test_scores_ab, test_scores_ba):
        test_score_map[tuple(b)] = (float(ab), float(ba))
    
    print("test score map",len(test_score_map) )
    #print("dev score map",len(dev_score_map) )
    
    curr_mentions = sorted(gold_map_test.keys())
    curr_gold_cluster_map = [(men, gold_map_test[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = working_folder + f'/probing_gold_{split}.keyfile'
    
    generate_key_file(curr_gold_cluster_map, 'evt', working_folder, gold_key_file)
    
    pairwise_scores = []
    for p in test_pairs:
        if tuple(p) in test_score_map:
            pairwise_scores.append(np.mean(test_score_map[p]))
    
    
    mid2cluster = cluster(curr_mentions, test_pairs, pairwise_scores, threshold=threshold) #checking if theoretically a perfect clustering can be achieved
    system_key_file = working_folder + f'/probing_system_{split}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', working_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)
    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    conf = np.round((mf + bf + cf) / 3, 1)
    print(working_folder, split)
    final_frame = [mr, mp, mf,br, bp, bf,cr, cp, cf,  lr, lp, lf,conf ]
    result_string = f'&& {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)
    return conf, result_string, final_frame

def get_probing_causal_counterpart_clusters_non_trainable(split,test_pairs, test_scores_ab, test_scores_ba, gold_map_test, working_folder, threshold = .5):

    split = 'test'
    test_score_map = {}
    for b, ab, ba in zip(test_pairs, test_scores_ab, test_scores_ba):
        test_score_map[tuple(b)] = (float(ab), float(ba))
    
    print("test score map",len(test_score_map) )
    #print("dev score map",len(dev_score_map) )
    
    curr_mentions = sorted(gold_map_test.keys())
    curr_gold_cluster_map = [(men, gold_map_test[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = working_folder + f'/probing_gold_{split}.keyfile'
    
    generate_key_file(curr_gold_cluster_map, 'evt', working_folder, gold_key_file)
    
    pairwise_scores = []
    for p in test_pairs:
        if tuple(p) in test_score_map:
            pairwise_scores.append(np.mean(test_score_map[p]))
    
    
    mid2cluster = cluster(curr_mentions, test_pairs, pairwise_scores, threshold=.5) #checking if theoretically a perfect clustering can be achieved
    system_key_file = working_folder + f'/probing_system_{split}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', working_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)
    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    conf = np.round((mf + bf + cf) / 3, 1)
    print(working_folder, split)
    final_frame = [mr, mp, mf,br, bp, bf,cr, cp, cf,  lr, lp, lf,conf ]
    result_string = f'&& {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)
    return conf, result_string, final_frame
def process_clustering_result(dataframe):

    # Define column names
    columns = ['Epoch', 'Metrics']
     # Convert the list of lists into a DataFrame
    df = pd.DataFrame(dataframe, columns=columns)

    # Separate the 'Metrics' column into individual columns
    df[['MUC R', 'MUC P', 'MUC F1', 'B3 R', 'B3 P', 'B3 F1', 'Ceafe R', 'Ceafe P', 'Ceafe F1', 'LEA R', 'LEA P', 'LEA F1', 'CoNLL F1']] = pd.DataFrame(df['Metrics'].tolist(), index=df.index)

    # Drop the original 'Metrics' column
    df.drop(columns=['Metrics'], inplace=True)

    # Set 'Epoch' column as index
    df.set_index('Epoch', inplace=True)

    return df

def accuracy(predicted_labels, true_labels):
    # Calculate the number of correct predictions
    correct_predictions = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
    # Calculate accuracy
    return correct_predictions / len(predicted_labels) if len(predicted_labels) > 0 else 0

def precision(predicted_labels, true_labels):
    true_positives = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true == 1)
    predicted_positives = sum(predicted_labels)
    return true_positives / predicted_positives if predicted_positives else 0

def recall(predicted_labels, true_labels):
    true_positives = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true == 1)
    actual_positives = sum(true_labels)
    return true_positives / actual_positives if actual_positives else 0
 
def f1_score(predicted_labels, true_labels):
    prec = precision(predicted_labels, true_labels)
    rec = recall(predicted_labels, true_labels)
    if prec + rec == 0:
        return 0  # Prevent division by zero
    return 2 * (prec * rec) / (prec + rec)


def train_dpos(dataset, model_name=None, trainable = True):

    final_coref = []
    final_f = []
    final_conf = []
    final_exp_results = []
    wtd = False
    # dataset = 'deli_data'
    if(dataset == 'wtd_dataset'):
            wtd = True
    print('This is the data', dataset)
    dataset_folder = f'./datasets/{dataset}/'
    device = torch.device('cuda:1')
    device_ids = list(range(1))
    device_ids = [1]
    #get the maps 
    probing_map, document_map = combine_non_probing_with_probing_map(dataset = dataset, wtd = wtd)
    #generate the train and dev sets here 
    #train_pairs, train_labels, dev_pairs, dev_labels, test_pairs, test_labels = generate_split_pairs_labels(dataset, probing_map, document_map, gold =False)
    #gold_map_train, gold_map_dev, gold_map_test = generate_gold_map(dataset = dataset)
    #print(len(train_pairs), len(train_labels),len (dev_pairs), len(dev_labels), len(test_pairs), len(test_labels)) # 2948 training positive pairs 
    #train_pairs_gold, _, dev_pairs_gold, _, test_pairs_gold, _ = generate_split_pairs_labels(dataset, probing_map, document_map, gold =False)
#     final_gold_map_dev = get_gold_cluster_map(dev_pairs_gold, probing_map, document_map)
#     final_gold_map_test = get_gold_cluster_map(test_pairs_gold, probing_map, document_map)
    
    # # resample all the splits to reflect the gold labels with proximity based causal counterpart labeling (with a probing utterance)
    # train_dict = dict(zip(train_pairs, train_labels))
    # gold_map_train = get_gold_map(train_dict, probing_map, document_map)
    # train_pairs_dict = resample_split(train_dict, gold_map_train)
    
    # dev_dict = dict(zip(dev_pairs, dev_labels))
    # gold_map_dev = get_gold_map(dev_dict, probing_map, document_map)
    # dev_pairs_dict = resample_split(dev_dict, gold_map_dev)
    test_cosine_scores_ab = []
    dev_cosine_scores_ab = []
    test_token_overlap_scores_ab = []
    dev_token_overlap_scores_ab = []
    test_iou_overlap_scores_ab = []
    dev_iou_overlap_scores_ab = []
    #dev_cosine_scores_ab = []
    
    # test_dict = dict(zip(test_pairs, test_labels))
    # gold_map_test = get_gold_map(test_dict, probing_map, document_map)


    gold_map_train, gold_map_dev, gold_map_test = generate_gold_map(dataset = dataset)
    
    ##STARTS HERE###
    # train_pairs, train_labels, causal_probing_label_train = generate_pairs_for_train_eval(gold_map_train, split='train')
    # #print("final pairs train", len(train_pairs), len(train_labels), len(causal_probing_label_train))
    # dev_pairs, dev_labels, causal_probing_label_dev = generate_pairs_for_train_eval(gold_map_dev, split = 'dev')
    # #print("final pairs dev", len(dev_pairs), len(dev_labels), len(causal_probing_label_dev)) 
    # test_pairs, test_labels, causal_probing_label_test = generate_pairs_for_train_eval(gold_map_test, split = 'test')
    # print(gold_map_test['d7439857-0e42-4d12-b54c-4593e6536ea4_5'])
    # # Filter to only include entries with the specific group_id
    # target_group_id = 'f8704b44-06e5-4938-8c61-6a64aabf442b'
    # filtered_data = [value for key, value in gold_map_train.items() if value['group_id'] == target_group_id]
    # import csv


    # # Determine the fields dynamically based on the entries in probing_map and document_map
    # fields_from_maps = ['original_text', 'annotation_type']
    # fields = ['m_id', 'gold_cluster', 'group_id', 'split'] + fields_from_maps + ['label']

    # # Append additional content from probing_map or document_map to each row and save back to CSV
    # with open('output_with_content.csv', 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fields)
    #     writer.writeheader()
    #     for row in filtered_data:
    #         m_id = row['m_id']
    #         if m_id in probing_map:
    #             map_data = probing_map[m_id]
    #         elif m_id in document_map:
    #             map_data = document_map[m_id]
    #         else:
    #             map_data = {key: 'N/A' for key in fields_from_maps}

    #         for key in fields_from_maps:
    #             row[key] = map_data.get(key, 'N/A')
    #         row['label'] = causal_probing_label_train.get(m_id, 'Label Not Available')
    #         writer.writerow(row)

    # print("CSV file has been updated with detailed content from the maps.")
    
     ##END HERE###

    train_pairs, train_labels, causal_probing_label_train = generate_pairs_for_train_eval(gold_map_train, split='train')
    #print("final pairs train", len(train_pairs), len(train_labels), len(causal_probing_label_train))
    dev_pairs, dev_labels, causal_probing_label_dev = generate_pairs_for_train_eval(gold_map_dev, split = 'dev')
    #print("final pairs dev", len(dev_pairs), len(dev_labels), len(causal_probing_label_dev)) 
    test_pairs, test_labels, causal_probing_label_test = generate_pairs_for_train_eval(gold_map_test, split = 'test')
    #print('GOTTEM',causal_probing_label_test['5418bfc6-123a-43fd-9d09-9846ec97d7cc'])
    
    #print("final pairs test", len(test_pairs), len(test_labels), len(causal_probing_label_test)) 
    print("pairs after gold map creation", len(train_pairs), len(train_labels),len (dev_pairs), len(dev_labels), len(test_pairs), len(test_labels)) # 2948 training positive pairs
    print("positive samples in train", sum([x for x in train_labels if x ==1]))
    print("neg samples in train", len(train_labels) -  sum([x for x in train_labels if x ==1]))
    print("positive samples in dev", sum([x for x in dev_labels if x ==1]))
    print("neg samples in dev", len(dev_labels) - sum([x for x in dev_labels if x ==1]))
    print("positive samples in test", sum([x for x in test_labels if x ==1]))
    print("neg samples in test", len(test_labels) -sum([x for x in test_labels if x ==1]))

    # for (m1, m2) in mention_pairs:
    #     #print(m1,m2)
    #     try: 

    #         sentence_a = str(probing_map[m1]['prev_utterance_history']) + "<m>" +  " " + probing_map[m1]['probing_utterance'] + " "  + "</m>"
    #     except KeyError:
    #         sentence_a = "<m>" + " " + str(document_map[m1]['origin']) + document_map[m1]['original_text'] + " " + "</m>"

    #     try:
    #         sentence_b = "<m>" + " " + str(document_map[m2]['origin']) + document_map[m2]['original_text'] + " " + "</m>" #get the causal counterpart utterance 
    #     except KeyError:
    #         sentence_b = str(probing_map[m2]['prev_utterance_history']) + "<m>" +  " " + probing_map[m2]['probing_utterance'] + " "  + "</m>"

    trainable = False
    if(trainable == False):
       # Get the threshold for the non-trainable baselines using the dev pairs
        for probing_id, message_id in tqdm(dev_pairs,desc="Generating thresholds for dev"): #For each dev pair (ProbingQuestionID, messageID)
            try: 
                probing_text = probing_map[probing_id]['probing_utterance'] #if probing_id in probing_map else 'Probing ID not found' #Probing Question
            except KeyError:
                probing_text = document_map[probing_id]['original_text']
            
            try:
                casual_text = document_map[message_id]['original_text'] #if message_id in document_map else 'Message ID not found' #Causal counterpart
            except KeyError:
                casual_text = probing_map[message_id]['probing_utterance']
            
            # Get embeddings
            probing_embedding = get_bert_embedding(probing_text)
            casual_embedding = get_bert_embedding(casual_text)
            #Get token overlap scores 
            overlap_score = calculate_fuzzy_score(probing_text, casual_text)
            dev_token_overlap_scores_ab.append(overlap_score)

            #get IOU scores
            iou_score = calculate_iou(probing_text, casual_text, wtd= True)
            dev_iou_overlap_scores_ab.append(iou_score)
            
            if probing_embedding is not None and casual_embedding is not None:
                sim = get_cosine_similarity(probing_embedding, casual_embedding) #Get cosine similarity
                dev_cosine_scores_ab.append(sim)

        overlap_threshold = (sum(dev_token_overlap_scores_ab) / len(dev_token_overlap_scores_ab))/100
        #dev_overlap_predictions = dev_token_overlap_scores_ab > overlap_threshold
        
        dev_overlap_predictions = [score > overlap_threshold for score in dev_token_overlap_scores_ab]

        cosine_threhsold = sum(dev_cosine_scores_ab)/ len(dev_cosine_scores_ab)
        #dev_cosine_predictions = dev_cosine_scores_ab > cosine_threhsold
        dev_cosine_predictions = [score > cosine_threhsold for score in dev_cosine_scores_ab]
        iou_threshold = (sum(dev_iou_overlap_scores_ab) / len(dev_iou_overlap_scores_ab))
        # cosine_threhsold =  0.5846265405609135

        # overlap_threshold = 0.29546056884292177
        
       
        
#         cosine_threhsold = 0.5846265405609135
#         overlap_threshold = 0.29546056884292177
        print('This is the cosine threshold ', cosine_threhsold)
        print('\nThis is the overlap threshold ', overlap_threshold)
        print('\nThis is the iou threshold ', iou_threshold)
        
        for probing_id, message_id in tqdm(test_pairs, desc="Generating scores for test"):
            try: 
                probing_text = probing_map[probing_id]['probing_utterance'] #if probing_id in probing_map else 'Probing ID not found' #Probing Question
            except KeyError:
                probing_text = document_map[probing_id]['original_text']
            
            try:
                casual_text = document_map[message_id]['original_text'] #if message_id in document_map else 'Message ID not found' #Causal counterpart
            except KeyError:
                casual_text = probing_map[message_id]['probing_utterance']
            
            # Get embeddings
            probing_embedding = get_bert_embedding(probing_text)
            casual_embedding = get_bert_embedding(casual_text)
            # Get token overlap 
            overlap_score = (calculate_fuzzy_score(probing_text, casual_text))/100
            test_token_overlap_scores_ab.append(overlap_score)
            test_iou = calculate_iou(probing_text,casual_text, wtd= True)
            test_iou_overlap_scores_ab.append(test_iou)
            
            if probing_embedding is not None and casual_embedding is not None:
                sim = get_cosine_similarity(probing_embedding, casual_embedding)
                test_cosine_scores_ab.append(sim)
        
        #print(test_cosine_scores_ab[0:20])
        
        #test_token_overlap_predictions = test_token_overlap_scores_ab > overlap_threshold
        test_token_overlap_predictions = [score > overlap_threshold for score in test_token_overlap_scores_ab]
        #test_cosine_overlap_predictions = test_cosine_scores_ab > cosine_threhsold
        test_cosine_predictions = [score > cosine_threhsold for score in test_cosine_scores_ab]

        test_iou_predictions = [score > iou_threshold for score in test_iou_overlap_scores_ab]
        #print(test_token_overlap_scores_ab[0:20])
        conf, final_scores, final_frame = get_probing_causal_counterpart_clusters_non_trainable("test",test_pairs, test_cosine_scores_ab, test_cosine_scores_ab, gold_map_test, dataset_folder, threshold= cosine_threhsold )        
        conf, final_scores, final_frame = get_probing_causal_counterpart_clusters_non_trainable("test",test_pairs, test_token_overlap_scores_ab, test_token_overlap_scores_ab, gold_map_test, dataset_folder, threshold= overlap_threshold) 
        conf, final_scores, final_frame = get_probing_causal_counterpart_clusters_non_trainable("test",test_pairs, test_iou_overlap_scores_ab, test_iou_overlap_scores_ab, gold_map_test, dataset_folder, threshold= iou_threshold) 
        
    print(len(test_pairs))
    pickle.dump(test_pairs, open('deli_test_pairs.pkl', 'wb'))    
    pickle.dump(test_cosine_scores_ab, open('deli_test_cosine_scores_ab.pkl', 'wb'))    
    pickle.dump(test_token_overlap_scores_ab, open('deli_test_token_overlap_scores_ab.pkl', 'wb'))    
    pickle.dump(test_iou_overlap_scores_ab, open('deli_test_iou_overlap_scores_ab.pkl', 'wb'))
    # pickle.dump(test_cosine_predictions, open('test_cosine_predictions.pkl', 'wb'))
    # pickle.dump(test_cosine_predictions, open('test_iou_overlap_scores_ab.pkl', 'wb'))
    # pickle.dump(test_iou_overlap_scores_ab, open('test_iou_overlap_scores_ab.pkl', 'wb'))
    # pickle.dump(test_iou_overlap_scores_ab, open('test_iou_overlap_scores_ab.pkl', 'wb'))    
    # pickle.dump(test_scores, open(test_sampling_folder + f'/{split}scores_ab{sampling}_{n}.pkl', 'wb'))
        
    return test_pairs, test_cosine_scores_ab, test_token_overlap_scores_ab, test_iou_overlap_scores_ab, test_cosine_predictions, test_token_overlap_predictions, test_iou_predictions, dev_cosine_scores_ab, dev_token_overlap_scores_ab, dev_cosine_predictions, dev_overlap_predictions
       
    #     #conf, final_scores, final_frame = get_probing_causal_counterpart_clusters("test",test_pairs, dev_scores_ab, dev_scores_ba, gold_map_test, working_folder )
    # print("dev accuracy:", accuracy(dev_predictions, dev_labels))
    #         print("dev precision:", precision(dev_predictions, dev_labels))
    #         print("dev recall:", recall(dev_predictions, dev_labels))
    #         print("dev f1:", f1_score(dev_predictions, dev_labels))
    # print("test accuracy:", accuracy(test_predictions, test_labels))
    #         print("test precision:", precision(test_predictions, test_labels))
    #         print("test recall:", recall(test_predictions, test_labels))
    #         print("test f1:", f1_score(test_predictions, test_labels))
    # # test_pairs_dict = resample_split(test_dict, gold_map_test)
    # print("gold map size", len(gold_map_train), len(gold_map_dev), len(gold_map_test))
    
    # train_pairs = list(train_pairs_dict.keys())
    # train_labels = list(train_pairs_dict.values())
    
    # dev_pairs = list(dev_pairs_dict.keys())
    # dev_labels = list(dev_pairs_dict.values())
    
    # test_pairs = list(test_pairs_dict.keys())
    # test_labels = list(test_pairs_dict.values())
    
    # print("positive samples in train", sum([y for x, y in train_pairs_dict.items() if y ==1]))
    # print("neg samples in train", len(train_pairs_dict) - sum([y for x, y in train_pairs_dict.items() if y ==1]))
    # print("positive samples in dev", sum([y for x, y in dev_pairs_dict.items() if y ==1]))
    # print("neg samples in train", len(dev_pairs_dict) - sum([y for x, y in dev_pairs_dict.items() if y ==1]))  
    # print("positive samples in test", sum([y for x, y in test_pairs_dict.items() if y ==1]))
    # print("neg samples in train", len(test_pairs_dict) - sum([y for x, y in test_pairs_dict.items() if y ==1]))
    
    #try a sample unit test with a smaller pos/neg set
    # train_pairs =train_pairs[0:50] + train_pairs[-50:] # 50 pos and 50 neg labels
    # train_labels = train_labels[0:50] + train_labels[-50:] 
    
#     # dev_pairs = dev_pairs[0:50] + dev_pairs[-50:]  # 50 pos and 50 neg labels
#     # dev_labels = dev_labels[0:50] + dev_labels[-50:]
    
#     # test_pairs = test_pairs[0:50] + test_pairs[-50:]  # 50 pos and 50 neg labels
#     # test_labels = dev_labels[0:50] + test_labels[-50:]
    
#     print("smaller prototype", len(train_pairs), len(train_labels),len (dev_pairs), len(dev_labels), len(test_pairs), len(test_labels)) # 2948 training positive pairs
    

#     # model_name = 'roberta-base'
#     scorer_module = CrossEncoder(is_training=True, long = True, model_name=model_name).to(device)

#     parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
#     parallel_model.module.to(device)
#     print("special tokens", parallel_model.module.start_id, parallel_model.module.end_id) 
    
#     print("sampling prototype if learning", len(train_pairs), len(train_labels))
    
#     conf, final_coref_results, final_coref_frame  = train(train_pairs, train_labels,dev_pairs, dev_labels, test_pairs, test_labels, \
#             parallel_model, probing_map,document_map,gold_map_dev,gold_map_test, dataset_folder, device,
#           batch_size=4, n_iters=16, lr_lm=0.000001, lr_class=0.0001,wtd= wtd) 
          
#     sampling_folder = dataset_folder + f'/probing_scores/sampling_prototype/'
#     final_exp_results.append(final_coref_frame)
    
#     columns = ['Epoch', 'MUC R', 'MUC P', 'MUC F1','B3 R', 'B3 P', 'B3 F1','Ceafe R', 'Ceafe P', 'Ceafe F1','LEA R', 'LEA P', 'LEA F1', 'CoNLL F1' ]
#     final_coref_df = pd.DataFrame(final_coref_results)
#     final_frame_df = process_clustering_result(conf)

#     final_frame_df.to_csv(sampling_folder + f"/{dataset}_sampling_scores.csv")
#     final_frame_df.to_csv(sampling_folder + f"/{dataset}_sampling_scores_df.csv")

#     return final_frame_df 

# def train(train_pairs, #USE THE DEV PAIRS AND DEV LABELS FROM HERE 
#           train_labels,
#           dev_pairs,
#           dev_labels,
#           test_pairs, 
#           test_labels,
#           parallel_model,
#           probing_map,
#           document_map,
#           gold_map_dev,
#           gold_map_test,
#           working_folder,
#           device,
#           batch_size=4,
#           n_iters=50,
#           lr_lm=0.00001,
#           lr_class=0.001,
#           wtd=False):
#     bce_loss = torch.nn.BCELoss()
#     # mse_loss = torch.nn.MSELoss()
    

#     optimizer = torch.optim.AdamW([
#         {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
#         {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
#     ])

#     # all_examples = load_easy_hard_data(trivial_non_trivial_path)
#     # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)
    
#     tokenizer = parallel_model.module.tokenizer

#     # prepare data
#     train_ab, train_ba = tokenize_utterances(tokenizer, train_pairs, probing_map, document_map, parallel_model.module.end_id,  max_sentence_len=512)
#     dev_ab, dev_ba = tokenize_utterances(tokenizer, dev_pairs, probing_map, document_map, parallel_model.module.end_id, max_sentence_len=512)
#     test_ab, test_ba = tokenize_utterances(tokenizer, test_pairs, probing_map, document_map, parallel_model.module.end_id, max_sentence_len=512)

#     # labels
#     train_labels = torch.FloatTensor(train_labels)
#     dev_labels = torch.LongTensor(dev_labels)
#     test_labels = torch.LongTensor(test_labels)
#     split = 'test'

#     final_coref_results = []
#     final_coref_frame = []
#     final_conf = []

#     for n in range(n_iters):
#         test_scores = []
#         train_indices = list(range(len(train_pairs)))
#         random.shuffle(train_indices)
#         iteration_loss = 0.
#         # new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
#         new_batch_size = batch_size
#         for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
#             optimizer.zero_grad()
#             batch_indices = train_indices[i: i + new_batch_size]

#             scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
#             scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices)

#             batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

#             scores_mean = (scores_ab + scores_ba) / 2

#             loss = bce_loss(scores_mean, batch_labels)

#             loss.backward()

#             optimizer.step()

#             iteration_loss += loss.item()
#         with torch.no_grad():
#             print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
#             # iteration accuracy
#             dev_scores_ab, dev_scores_ba = predict_causal_counterpart(parallel_model, dev_ab, dev_ba, device, batch_size)
#             dev_predictions = (dev_scores_ab + dev_scores_ba)/2
#             dev_predictions = dev_predictions > 0.5
#             dev_predictions = torch.squeeze(dev_predictions)

#             print("dev accuracy:", accuracy(dev_predictions, dev_labels))
#             print("dev precision:", precision(dev_predictions, dev_labels))
#             print("dev recall:", recall(dev_predictions, dev_labels))
#             print("dev f1:", f1_score(dev_predictions, dev_labels))

#             test_scores_ab, test_scores_ba = predict_causal_counterpart(parallel_model, test_ab, test_ba, device, batch_size)
#             test_predictions = (test_scores_ab + test_scores_ba)/2
#             test_scores.append(test_predictions)
#             test_predictions = test_predictions > 0.5
#             test_predictions = torch.squeeze(test_predictions)

#             print("test accuracy:", accuracy(test_predictions, test_labels))
#             print("test precision:", precision(test_predictions, test_labels))
#             print("test recall:", recall(test_predictions, test_labels))
#             print("test f1:", f1_score(test_predictions, test_labels))

#             # get dev scores 
#             #conf, final_scores, final_frame = get_probing_causal_counterpart_clusters(split= "dev",test_pairs, test_scores_ab, test_scores_ba, gold_map_test, working_folder )
          
#             #Create a new functions that miminc get_probing_causal_counterpart_clusters. Send test_scores_ab, test_scores_ab, gold_map_test
#             # get test scores 
#             conf, final_scores, final_frame = get_probing_causal_counterpart_clusters("test",dev_pairs, dev_scores_ab, dev_scores_ba, gold_map_dev, working_folder )
#             final_coref_results.append([n,final_scores ])
#             final_coref_frame.append([n,conf])
#             final_conf.append([n,final_frame])
#             pickle.dump(test_scores, open(working_folder + f'/probing_scores/sampling_prototype/{split}_scores_ab_{n}.pkl', 'wb'))
#             pickle.dump(test_pairs, open(working_folder + f'/probing_scores/sampling_prototype/{split}_pairs.pkl', 'wb'))

            
#         if n % 2 == 0:
#             scorer_folder = working_folder + f'/probing_scorer/chk_{n}'
#             if not os.path.exists(scorer_folder):
#                 os.makedirs(scorer_folder)
#             model_path = scorer_folder + '/linear.chkpt'
#             torch.save(parallel_model.module.linear.state_dict(), model_path)
#             parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
#             parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
#             print(f'saved model at {n}')

#     scorer_folder = working_folder + '/probing_scorer/'
#     if not os.path.exists(scorer_folder):
#         os.makedirs(scorer_folder)
#     model_path = scorer_folder + '/linear.chkpt'
#     torch.save(parallel_model.module.linear.state_dict(), model_path)
#     parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
#     parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
#     pickle.dump(test_scores, open(working_folder + f'/probing_scores/sampling_prototype/{split}_scores_ab_{n}.pkl', 'wb'))
#     pickle.dump(test_pairs, open(working_folder + f'/probing_scores/sampling_prototype/{split}_pairs.pkl', 'wb'))

#     return final_conf, final_coref_results, final_coref_frame

if __name__ == '__main__':
#     if len(sys.argv) < 2: 
#         print("Usage: python probing_training_clustering.py deli_data/wtd_dataset") 
#         sys.exit(1) 
#     dataset = sys.argv[1]
    #trainable = sys.argv[2]
    dataset = 'deli_data'
    trainable = False
    gold_map_test, test_cosine_scores_ab, test_token_overlap_scores_ab, test_iou_overlap_scores_ab, test_cosine_predictions, test_token_overlap_predictions, test_iou_predictions, dev_cosine_scores_ab, dev_token_overlap_scores_ab, dev_cosine_predictions, dev_overlap_predictions = train_dpos(dataset, model_name='allenai/longformer-base-4096', trainable = trainable)
    
    # pickle.dump(gold_map_test, open( f'scores_ab.pkl', 'wb'))
    # pickle.dump(test_pairs, open(test_sampling_folder + f'/{split}_pairs.pkl', 'wb'))
    
    data = {
    #"gold_map_test": gold_map_test,
    "test_cosine_scores_ab": test_cosine_scores_ab,
    "test_token_overlap_scores_ab": test_token_overlap_scores_ab,
    "test_iou_overlap_scores_ab": test_iou_overlap_scores_ab,
    "test_cosine_predictions": test_cosine_predictions,
    "test_token_overlap_predictions": test_token_overlap_predictions,
    "test_iou_predictions": test_iou_predictions
    # "dev_cosine_scores_ab": dev_cosine_scores_ab,
    # "dev_token_overlap_scores_ab": dev_token_overlap_scores_ab,
    # "dev_cosine_predictions": dev_cosine_predictions,
    # "dev_overlap_predictions": dev_overlap_predictions
    }
    print(gold_map_test)
    print(len(gold_map_test))
    print(len(test_cosine_scores_ab), len(test_token_overlap_scores_ab),len(test_iou_overlap_scores_ab),len(test_cosine_predictions),len(test_token_overlap_predictions),len(test_iou_predictions),len(dev_cosine_scores_ab))
    df = pd.DataFrame(data)

    # Save DataFrame to a CSV file
    csv_path = "deli_Detailed_Model_Results.csv"
    df.to_csv(csv_path, index=False)    
    # pickle.dump(test_scores, open(test_sampling_folder + f'/{split}scores_ab{sampling}_{n}.pkl', 'wb'))
    # pickle.dump(test_pairs, open(test_sampling_folder + f'/{split}_pairs.pkl', 'wb'))