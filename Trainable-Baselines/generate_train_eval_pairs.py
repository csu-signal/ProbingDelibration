import pickle
import torch
from helper_probing import tokenize, f1_score, accuracy, precision, recall, cluster, generate_key_file
from helper_testing import tokenize_utterances, forward_ab, tokenize_wtd_utterances
from probing_prediction import predict_causal_counterpart
import random
from tqdm import tqdm
import os
# from modeling_probing import Causal_Intervention_Scorer
from generate_gold_map_wtd import generate_gold_map
from delitoolkit.delidata import DeliData
import pickle
import json 
import pandas as pd
import numpy as np
import heapq
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
 

from collections import defaultdict
 


def generate_pairs_for_train_eval(gold_map, utterance_sequence_map, split, previous_window = None):
    
    '''
    Generates all antecedent pairs and binary labels of casual and probing interventions using the gold labels
    generated with "get_gold_map_cldeaned".
    Outputs pairs, labels and causal and probing label for each train/eval pair
     
    '''
   
    
#     utterance_sequence_map = utterance_seq_dict
    group_to_interventions = defaultdict(list)
    pairs_labels_dict = {}
    pair_sample = []
    split_intervention_ids = sorted([m_id for m_id, m in gold_map.items() if m['split'] == split])
    zero_list = []
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
        for i in range(len(list_interventions)): #mid and pid
            for j in range(i + 1):
                if i != j:
                    #find diff 
                    d1 = int(utterance_sequence_map[list_interventions[i]]['utterance_id'])
                    d2 = int(utterance_sequence_map[list_interventions[j]]['utterance_id'])
                    diff = abs(d2-d1)
                    if diff < previous_window and diff==0:
                        zero_list.append((list_interventions[i], list_interventions[j],int(gold_map[list_interventions[i]]['gold_cluster'] ==gold_map[list_interventions[j]]['gold_cluster']) ))
                    
                    if diff <= previous_window: # IMP: Removing zero pairs removes intervention pairs that form true clusters. Small number but affects system performance. Keeping these pairs for now to be consistent with the gold clustering for now. 
                     
                        intervention_pairs.append((list_interventions[i], list_interventions[j])) # add the pair label and the causal label 

                        intervention_labels.append(int(gold_map[list_interventions[i]]['gold_cluster'] ==gold_map[list_interventions[j]]['gold_cluster']) )
                 
    # add the pair label and the causal label 
    for intervention in split_intervention_ids:
        if len(intervention.split("_")) > 1:
            causal_probing_label[intervention] = 'probing'
        else:
            causal_probing_label[intervention] = 'causal'
    print(f" {split} final pairs", len(intervention_pairs), len(intervention_labels), len(causal_probing_label))
    return intervention_pairs, intervention_labels, causal_probing_label, zero_list    

dataset = 'wtd_dataset'
dataset_folder = f'./datasets/{dataset}/'
utterance_seq_dict_file = f'{dataset_folder}/wtd_with_utterance_2.pkl'
with open(utterance_seq_dict_file, "rb") as f:
    utterance_sequence_map  = pickle.load(f)
gold_map_train, gold_map_dev, gold_map_test, probing_map, document_map = generate_gold_map(dataset = 'wtd_dataset')
 

top_k = list(range(22, 70, 1)) # just checking at what k window we can get all pos pairs, but its pretty long
# for k in top_k:
# print("lk",k)
train_pairs, train_labels, causal_probing_label_train, zero_train = generate_pairs_for_train_eval(gold_map_train,utterance_sequence_map, split='train', previous_window =60)
#print("final pairs train", len(train_pairs), len(train_labels), len(causal_probing_label_train))
dev_pairs, dev_labels, causal_probing_label_dev, zero_dev = generate_pairs_for_train_eval(gold_map_dev,utterance_sequence_map, split = 'dev', previous_window =60)
#print("final pairs dev", len(dev_pairs), len(dev_labels), len(causal_probing_label_dev)) 
test_pairs, test_labels, causal_probing_label_test, zero_test = generate_pairs_for_train_eval(gold_map_test,utterance_sequence_map, split = 'test', previous_window =60)
print("POS", sum(x for x in train_labels if x ==1), "NEG",  len(train_labels)-sum(x for x in train_labels if x ==1))
print("POS", sum(x for x in dev_labels if x ==1),"NEG", len(dev_labels)-sum(x for x in dev_labels if x ==1))
print("POS", sum(x for x in test_labels if x ==1),"NEG", len(test_labels)-sum(x for x in test_labels if x ==1))

#gold_map_train, gold_map_dev, gold_map_test, probing_map, document_map = generate_gold_map(dataset = 'deli_data')
