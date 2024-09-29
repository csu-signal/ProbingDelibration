import pickle
import random
from tqdm import tqdm
import os
from delitoolkit.delidata import DeliData
import pickle
import json 
import pandas as pd
import numpy as np
from collections import defaultdict


def find_extreme_item(p_ids_set, map_probing_with_min_causal=True):
    # Initialize variables to keep track of extreme item
    extreme_item = None
    extreme_value = None if map_probing_with_min_causal else float('-inf')  # Initialize to negative infinity for largest

    # Iterate over each item in the set
    for item in p_ids_set:
        # Split the item by underscore
        parts = item.split('_')
        # Extract the value after the underscore and convert it to an integer
        value = int(parts[-1])
        # Check if the current value is extreme based on the flag
        if map_probing_with_min_causal:
            if extreme_value is None or value < extreme_value:
                extreme_value = value
                extreme_item = item
        else:
            if value > extreme_value:
                extreme_value = value
                extreme_item = item

    return extreme_item

def get_gold_map_cleaned(all_dict,probing_map, document_map, map_probing_with_min_causal = True):

    '''
    Generates gold clusters from the utterances mapping every causal and probing intervention
    that form a deliberation chain to a gold label
    For both types of interventions, unique chains are given the earliest p_id as the gold label
    For m_ids with multiple sequential p_ids, gold clusters are assigned to the earliest p_id
    Bug fixed: no singletons are created. Leftover m_ids are mapped back to the earliest p_id in the chain 
    Returns the gold map:
    Signature :  gold_map[m_id or p_id]  = {'m_id':##, 'gold_cluster':##, 'group_id': ##}

    '''
    
    map_probing_with_min_causal = True

    m_id_occurrences = defaultdict(list)
    gold_cluster_map = {}
    m_id_to_gold_cluster_map = {}
    final_gold_map_new = {}
    all_probing_for_m_id = defaultdict(list)
    gold_map_cleaned = {}
    temp_map = defaultdict(dict)
    primary_keys_by_gold_cluster = defaultdict(list)
    # Iterate through the gold_cluster_map dictionary
    p_id_occurrences = defaultdict(list)
    for x, info in all_dict.items():
        m_id_occurrences[x[1]].append(x[0])
        p_id_occurrences[x[0]].append(x[1])
       
    duplicate_x_keys = [x_values for m_id, x_values in m_id_occurrences.items() if len(x_values) > 1]

    for  m_id, x_values in m_id_occurrences.items():

        if len(x_values) > 1:
            duplicate_x_keys = x_values
            all_probing_for_m_id[m_id].append(duplicate_x_keys)
            min_elements = []
            min_element = None
            min_digit = float('inf')
            max_element = None
            max_elements = []
            max_digit = float('-inf')
            probing_elements = []
            # Iterate through each sublist in the data list
            for item in duplicate_x_keys:
                # Split the element by underscores
                parts = item.split('_')
                # Extract the digit after the underscore
                digit = int(parts[-1])
                if map_probing_with_min_causal:
                # Check if the digit is smaller than the current minimum
                    if digit < min_digit:
                        #print("min digit", min_digit, m_id, item)
                        min_digit = digit
                        #print("min digit assign", min_digit, m_id, item)
                        min_element = item

                    # Append the minimum element to the result list
                        min_elements.append(min_element)
                        #print(min_element)
                        m_id_to_gold_cluster_map[m_id] = min_element  # this maps a causal statement to earliest in discourse 
                else:
                    if digit > max_digit:
                        #print("max probing to causal ")
                        max_digit = digit
                        max_element = item
                        max_elements.append(max_element)
                        m_id_to_gold_cluster_map[m_id] = max_element # this maps a causal statement to furthest in discourse 
                        probing_elements.append({'m_id': m_id, 'p_id': max_element})

                all_probing_for_m_id[m_id] = probing_elements
                
    
    
    
    
    for  m_id, x_values in m_id_occurrences.items():
        if m_id in m_id_to_gold_cluster_map:
            
            gold_map_cleaned[m_id] = {'m_id':m_id, 'gold_cluster':m_id_to_gold_cluster_map[m_id], 'group_id': document_map[m_id]['group_id']}
            #gold_map_cleaned[m_id] = m_id_to_gold_cluster_map[m_id] # this sets the min or max pid as the gold label
        else:
            gold_map_cleaned[m_id]  = {'m_id':m_id, 'gold_cluster':m_id_occurrences[m_id][0], 'group_id': document_map[m_id]['group_id']}
            #gold_map_cleaned[m_id] = m_id_occurrences[m_id][0]

    for  m_id, x_values in m_id_occurrences.items():
        #for p_id in x_values:
        p_ids_set = set(x_values)

        #print(x_values)
        extreme_p_id = find_extreme_item(p_ids_set, map_probing_with_min_causal=True)
        #print("exteme pid", extreme_p_id)
        for p_id in x_values:
            if len(p_ids_set) >1:
                #gold_map_cleaned[p_id] =  extreme_p_id
                if p_id in temp_map.keys():
                    #print(" > 1 sample continue this loop for new ", p_id,temp_map[p_id] )
                    continue
                else:
                    
                    temp_map[p_id]  = {'m_id':p_id, 'gold_cluster':extreme_p_id, 'group_id': document_map[m_id]['group_id']}
                    #print("> 1 sample this loop for new ", p_id, extreme_p_id,temp_map[p_id] )
                #temp_map[p_id] = extreme_p_id
            else:
                if p_id in temp_map.keys():
                    #print("one sample loop continue this loop for new ", p_id,temp_map[p_id] )
                    continue 
                else:
                    temp_map[p_id]  = {'m_id':p_id, 'gold_cluster':extreme_p_id, 'group_id': document_map[m_id]['group_id']}
                    #print(" one sample loop this loop for new ", p_id, extreme_p_id, temp_map[p_id])
                    #temp_map[p_id] = extreme_p_id


    gold_map_cleaned.update(temp_map)
    for  m_id, x_values in m_id_occurrences.items():
        if len(x_values) ==1:
            #print(x_values)
            p_id= x_values[0]
            #print("pid", p_id)
            if gold_map_cleaned[m_id] != gold_map_cleaned[p_id]: 
                #print(gold_map_cleaned[m_id],gold_map_cleaned[p_id] )
                gold_map_cleaned[m_id] = gold_map_cleaned[p_id]
    
    
    
    # Iterate over the dictionary items
    for primary_key, data in gold_map_cleaned.items():
        gold_cluster = data.get('gold_cluster')
        group_id = data.get('group_id')
        if gold_cluster:
            # Append the primary key to the list corresponding to the gold_cluster value
            primary_keys_by_gold_cluster[gold_cluster].append({'primary_key': primary_key, 'group_id': group_id, 'm_id': data['m_id']})
    
    
    singleton_m_ids = [ y[0]['primary_key'] for x, y in primary_keys_by_gold_cluster.items() if len(y) ==1]
#     for singleton in singleton_m_ids:
        
    
    for singleton, p_ids in m_id_occurrences.items():
        if singleton in singleton_m_ids:
            singleton_set = set(p_ids)
            extreme_p_id = find_extreme_item(singleton_set, map_probing_with_min_causal=True)
            #get the earlier cluster of the minimal pid 
            earlier_cluster = gold_map_cleaned[extreme_p_id]['gold_cluster']
            #gold_map_cleaned[singleton]['gold_cluster'] = extreme_p_id
            gold_map_cleaned[singleton] = {'m_id':singleton, 'gold_cluster':earlier_cluster, 'group_id': document_map[singleton]['group_id']}

    
    
    return gold_map_cleaned, m_id_occurrences, p_id_occurrences

def get_gold_map_cleaned_old(all_dict,probing_map, document_map, map_probing_with_min_causal = True):
    '''
    Generates gold clusters from the utterances mapping every causal and probing intervention
    that form a deliberation chain to a gold label
    For both types of interventions, unique chains are given the earliest p_id as the gold label
    For m_ids with multiple sequential p_ids, gold clusters are assigned to the earliest p_id
    Bug: Singletons are created that consist of causal intervention (m_id) that has one unique probing intervention (p_id) 
    '''
    
    map_probing_with_min_causal = True

    m_id_occurrences = defaultdict(list)
    gold_cluster_map = {}
    m_id_to_gold_cluster_map = {}
    final_gold_map_new = {}
    all_probing_for_m_id = defaultdict(list)
    gold_map_cleaned = {}
    temp_map = {}
    # Iterate through the gold_cluster_map dictionary
    p_id_occurrences = defaultdict(list)
    for x, info in all_dict.items():
        m_id_occurrences[x[1]].append(x[0])
        p_id_occurrences[x[0]].append(x[1])
       
    duplicate_x_keys = [x_values for m_id, x_values in m_id_occurrences.items() if len(x_values) > 1]

    for  m_id, x_values in m_id_occurrences.items():

        if len(x_values) > 1:
            duplicate_x_keys = x_values
            all_probing_for_m_id[m_id].append(duplicate_x_keys)
            min_elements = []
            min_element = None
            min_digit = float('inf')
            max_element = None
            max_elements = []
            max_digit = float('-inf')
            probing_elements = []
            # Iterate through each sublist in the data list
            for item in duplicate_x_keys:
                # Split the element by underscores
                parts = item.split('_')
                # Extract the digit after the underscore
                digit = int(parts[-1])
                if map_probing_with_min_causal:
                # Check if the digit is smaller than the current minimum
                    if digit < min_digit:
                        #print("min digit", min_digit, m_id, item)
                        min_digit = digit
                        #print("min digit assign", min_digit, m_id, item)
                        min_element = item

                    # Append the minimum element to the result list
                        min_elements.append(min_element)
                        #print(min_element)
                        m_id_to_gold_cluster_map[m_id] = min_element  # this maps a causal statement to earliest in discourse 
                else:
                    if digit > max_digit:
                        #print("max probing to causal ")
                        max_digit = digit
                        max_element = item
                        max_elements.append(max_element)
                        m_id_to_gold_cluster_map[m_id] = max_element # this maps a causal statement to furthest in discourse 
                        probing_elements.append({'m_id': m_id, 'p_id': max_element})

                all_probing_for_m_id[m_id] = probing_elements
                
                
    for  m_id, x_values in m_id_occurrences.items():
        if m_id in m_id_to_gold_cluster_map:
            
            gold_map_cleaned[m_id] = {'m_id':m_id, 'gold_cluster':m_id_to_gold_cluster_map[m_id], 'group_id': document_map[m_id]['group_id']}
            #gold_map_cleaned[m_id] = m_id_to_gold_cluster_map[m_id] # this sets the min or max pid as the gold label
        else:
            gold_map_cleaned[m_id]  = {'m_id':m_id, 'gold_cluster':m_id_occurrences[m_id][0], 'group_id': document_map[m_id]['group_id']}
            #gold_map_cleaned[m_id] = m_id_occurrences[m_id][0]

    for  m_id, x_values in m_id_occurrences.items():
        #for p_id in x_values:
        p_ids_set = set(x_values)

        #print(x_values)
        extreme_p_id = find_extreme_item(p_ids_set, map_probing_with_min_causal=True)
        #print("exteme pid", extreme_p_id)
        for p_id in x_values:
            if len(p_ids_set) >1:
                #gold_map_cleaned[p_id] =  extreme_p_id
                temp_map[p_id]  = {'m_id':p_id, 'gold_cluster':extreme_p_id, 'group_id': document_map[m_id]['group_id']}
                #temp_map[p_id] = extreme_p_id
            else:
                if p_id in temp_map.keys():
                    continue 
                else:
                    temp_map[p_id]  = {'m_id':p_id, 'gold_cluster':extreme_p_id, 'group_id': document_map[m_id]['group_id']}
                    #temp_map[p_id] = extreme_p_id


    gold_map_cleaned.update(temp_map)
    for  m_id, x_values in m_id_occurrences.items():
        if len(x_values) ==1:
            #print(x_values)
            p_id= x_values[0]
            #print("pid", p_id)
            if gold_map_cleaned[m_id] != gold_map_cleaned[p_id]: 
                #print(gold_map_cleaned[m_id],gold_map_cleaned[p_id] )
                gold_map_cleaned[m_id] = gold_map_cleaned[p_id]
                
    return gold_map_cleaned


def get_gold_cluster_sizes(gold_map, split=None):
    primary_keys_by_gold_cluster = defaultdict(list)
    # Iterate over the dictionary items
    for primary_key, data in gold_map.items():
        gold_cluster = data.get('gold_cluster')
        group_id = data.get('group_id')
        if gold_cluster:
            # Append the primary key to the list corresponding to the gold_cluster value
            primary_keys_by_gold_cluster[gold_cluster].append({'primary_key': primary_key, 'group_id': group_id, 'm_id': data['m_id']})
            
    # Print the dictionary containing primary keys for each gold_cluster value
    cluster_sizes = {}
    for gold_cluster, primary_keys in primary_keys_by_gold_cluster.items():
        cluster_sizes[gold_cluster] = len(primary_keys)
        
#         for key_group_pair in primary_keys:
#             print(f"Primary Key: {key_group_pair['primary_key']}, Group ID: {key_group_pair['group_id']}")
    
    # Calculate minimum size
    min_size = min(cluster_sizes.values())

    # Calculate maximum size
    max_size = max(cluster_sizes.values())

    # Calculate average size
    average_size = sum(cluster_sizes.values()) / len(cluster_sizes)
    
    # Count number of singletons
    singletons = [size for size in cluster_sizes.values() if size == 1]
    print(f"Number of singletons for {split}: {len(singletons)}")
    print(f"Minimum size for {split}: {min_size}")
    print(f"Maximum size for {split}: {max_size}")
    print(f"Average size for {split}: {average_size}")
    print(f"Total number of clusters for {split}: {len(primary_keys_by_gold_cluster)}")

    return primary_keys_by_gold_cluster, cluster_sizes





def get_gold_map(test_dict, probing_map, document_map, map_probing_with_min_causal = True):
    
    m_id_occurrences = defaultdict(list)
    gold_cluster_map = {}
    m_id_to_gold_cluster_map = {}
    final_gold_map_new = {}
    all_probing_for_m_id = defaultdict(list)
    # Iterate through the gold_cluster_map dictionary

    for x, info in test_dict.items():
        m_id_occurrences[x[1]].append(x[0])

    # Find primary keys (x) where the 'm_id' value occurs more than once
    duplicate_x_keys = [x_values for m_id, x_values in m_id_occurrences.items() if len(x_values) > 1]

    for  m_id, x_values in m_id_occurrences.items():

        if len(x_values) > 1:
            duplicate_x_keys = x_values
            all_probing_for_m_id[m_id].append(duplicate_x_keys)
            min_elements = []
            min_element = None
            min_digit = float('inf')
            max_element = None
            max_elements = []
            max_digit = float('-inf')
            probing_elements = []
            # Iterate through each sublist in the data list
            for item in duplicate_x_keys:
                # Split the element by underscores
                parts = item.split('_')
                # Extract the digit after the underscore
                digit = int(parts[-1])
                if map_probing_with_min_causal:
                # Check if the digit is smaller than the current minimum
                    if digit < min_digit:
                        #print("min digit", min_digit, m_id, item)
                        min_digit = digit
                        #print("min digit assign", min_digit, m_id, item)
                        min_element = item

                    # Append the minimum element to the result list
                        min_elements.append(min_element)
                        #print(min_element)
                        m_id_to_gold_cluster_map[m_id] = min_element  # this maps a causal statement to earliest in discourse 
                else:
                    if digit > max_digit:
                        #print("max probing to causal ")
                        max_digit = digit
                        max_element = item
                        max_elements.append(max_element)
                        m_id_to_gold_cluster_map[m_id] = max_element # this maps a causal statement to furthest in discourse 
                        probing_elements.append({'m_id': m_id, 'p_id': max_element})
                    
                all_probing_for_m_id[m_id] = probing_elements
   
    for index, (x,y) in enumerate(test_dict.items()):
        #print(index, x, y)
       
        if x[1] in m_id_to_gold_cluster_map.keys():
            #print(y, x)

            #print("this loop is getting", m_id_to_gold_cluster_map[x[1]] ,x[1], x[0])
            final_gold_map_new[x[1]] = {'m_id':x[1], 'gold_cluster': m_id_to_gold_cluster_map[x[1]], 'group_id': document_map[x[1]]['group_id']}
        else:
            #print(x[0], x[1], "else loop")

            final_gold_map_new[x[1]] = {'m_id':x[1], 'gold_cluster': x[0], 'group_id': document_map[x[1]]['group_id']}
        if x[0] in m_id_to_gold_cluster_map.values():
            final_gold_map_new[x[0]] = {'m_id':x[1], 'gold_cluster': x[0], 'group_id': document_map[x[1]]['group_id']}
            #print("other prpbing questions iutside ", x[0])
        else:
            
            
            for m_id, probing_elements in all_probing_for_m_id.items():
                p_ids_set = set(element['p_id'] for element in probing_elements)
                if x[0] in p_ids_set:
                    extreme_item = find_extreme_item(p_ids_set, map_probing_with_min_causal=map_probing_with_min_causal)
                    #print("yes pid set probing and max or min item", x[0], extreme_item)
                    final_gold_map_new[x[0]] = {'m_id':x[1], 'gold_cluster':  extreme_item, 'group_id': document_map[x[1]]['group_id']}
                    
        #                     print(f"m_id: {m_id}, p_ids_set: {p_ids_set}")
#                 else:
#                    #rint("no pid found anywhere", x[0])
        #print("right before end loop", x[0])
        if x[0] in final_gold_map_new.keys():
            continue
        else:
            final_gold_map_new[x[0]] = {'m_id':x[1], 'gold_cluster': x[0], 'group_id': document_map[x[1]]['group_id']}
        
                 
    return final_gold_map_new, all_probing_for_m_id, m_id_to_gold_cluster_map



def resample_split(test_dict, gold_map): 

    total_correct_gold = []
    total_correct_negative = []
    new_test_pairs_dict = {}
    negative_pairs_labels = {x: y for x, y in test_dict.items() if y ==0}
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
#         else:
#             print("negative split in resampling")
#             gold_probing = gold_map[x[0]]['gold_cluster']
#             gold_cc = gold_map[x[1]]['gold_cluster']
#             if gold_probing != gold_cc:
#                 total_correct_negative.append(1)
#                 new_test_pairs_dict[x] = y
#             else:
#                 new_test_pairs_dict[x] = 1
    new_test_pairs_dict.update(negative_pairs_labels)
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


#         for index, (x, y) in enumerate(negative_pairs_cleaned): # get the negative pairwise labels as zero 
#             if causal_counterpart_map[x]['set'] =="Train":
#                 train_pairs.append((x,y))
#                 train_labels.append(0)
    return train_pairs, train_labels, dev_pairs, dev_labels, test_pairs, test_labels


def combine_non_probing_with_probing_map(dataset):
    
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
    
    
    for group, messages in delidata_corpus.corpus.items():
        utternace_id = 1
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
                                'sol_tracker_all': m['sol_tracker_all'],
                    'utterance_id': utternace_id,
                        


        #                         'prev_uttterance_history': prev_history_map[key]
                            }
                utternace_id = utternace_id+1
    wtd = False
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


def generate_gold_map(dataset):
    
    dataset =  dataset
    probing_map, document_map = combine_non_probing_with_probing_map(dataset = dataset)


    train_pairs, train_labels, dev_pairs, dev_labels, test_pairs, test_labels = generate_split_pairs_labels(dataset, probing_map, document_map, gold =False)
    print(len(train_pairs), len(train_labels),len (dev_pairs), len(dev_labels), len(test_pairs), len(test_labels)) # 2948 training positive pairs 
    train_pairs_gold, train_labels_gold, dev_pairs_gold, dev_labels_gold, test_pairs_gold, test_labels_gold = generate_split_pairs_labels(dataset, probing_map, document_map, gold =True )
    #     final_gold_map_dev = get_gold_cluster_map(dev_pairs_gold, probing_map, document_map)
    #     final_gold_map_test = get_gold_cluster_map(test_pairs_gold, probing_map, document_map)

    # resample all the splits to reflect the gold labels with proximity based causal counterpart labeling (with a probing utterance)
    all_train_dict =  dict(zip(train_pairs, train_labels))
    all_dev_dict =  dict(zip(dev_pairs, dev_labels))
    all_test_dict = dict(zip(test_pairs, test_labels))

    train_dict = dict(zip(train_pairs_gold, train_labels_gold))

    gold_map_train, _, _  = get_gold_map_cleaned(train_dict, probing_map, document_map, map_probing_with_min_causal = False)


    train_pairs_dict = resample_split(all_train_dict, gold_map_train)

    dev_dict = dict(zip(dev_pairs_gold, dev_labels_gold))

    gold_map_dev, _, _ = get_gold_map_cleaned(dev_dict, probing_map, document_map, map_probing_with_min_causal = False)
    dev_pairs_dict = resample_split(all_dev_dict, gold_map_dev)

    test_dict = dict(zip(test_pairs_gold, test_labels_gold))

    gold_map_test,_, _ = get_gold_map_cleaned(test_dict, probing_map, document_map, map_probing_with_min_causal = False)
    test_pairs_dict = resample_split(all_test_dict, gold_map_test)


    print("gold map size", len(gold_map_train), len(gold_map_dev), len(gold_map_test))
    #print("after th elop", gold_map_test['8a93c6ef-41bc-4ecd-9cf0-aa83c7dc6f7c_3'])  # check this key for debugging 

    train_pairs = list(train_pairs_dict.keys())
    train_labels = list(train_pairs_dict.values())

    dev_pairs = list(dev_pairs_dict.keys())
    dev_labels = list(dev_pairs_dict.values())

    test_pairs = list(test_pairs_dict.keys())
    test_labels = list(test_pairs_dict.values())

    print("positive samples in train", sum([y for x, y in train_pairs_dict.items() if y ==1]))
    print("neg samples in train", len(train_pairs_dict) - sum([y for x, y in train_pairs_dict.items() if y ==1]))
    print("positive samples in dev", sum([y for x, y in dev_pairs_dict.items() if y ==1]))
    print("neg samples in train", len(dev_pairs_dict) - sum([y for x, y in dev_pairs_dict.items() if y ==1]))  
    print("positive samples in test", sum([y for x, y in test_pairs_dict.items() if y ==1]))
    print("neg samples in train", len(test_pairs_dict) - sum([y for x, y in test_pairs_dict.items() if y ==1]))

    train_cluster, train_cluster_sizes =  get_gold_cluster_sizes(gold_map_train, split = "train")
    dev_cluster, dev_cluster_sizes = get_gold_cluster_sizes(gold_map_dev, split = "dev")
    test_cluster, test_cluster_sizes = get_gold_cluster_sizes(gold_map_test, split = "test")
    singletons = [size for size in train_cluster_sizes.values() if size == 1]

    # add sploit information

    for intervention_id, gold_label in gold_map_train.items():
        gold_map_train[intervention_id]['split']='train'
    for intervention_id, gold_label in gold_map_dev.items():
        gold_map_dev[intervention_id]['split']='dev'
    for intervention_id, gold_label in gold_map_test.items():
        gold_map_test[intervention_id]['split']='test'

    return gold_map_train, gold_map_dev, gold_map_test

def generate_pairs_for_train_eval(gold_map, split, previous_window = None):
   
    utterance_seq_dict_file = '/s/babbage/b/nobackup/nblancha/public-datasets/ilideep/Probing_Deliberation/Data/deli_with_utterance.pkl'
    with open(utterance_seq_dict_file, "rb") as f:
        utterance_sequence_map  = pickle.load(f)
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
    #print(utterance_sequence_map)
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


import sys 
if __name__ == '__main__':
    top_k = list(range(5, 100, 1))
    
    gold_map_train, gold_map_dev, gold_map_test = generate_gold_map(dataset = 'deli_data')
    for k in top_k:
        print("lk",k)
        if(k>35):
            break

        train_pairs, train_labels, causal_probing_label_train, zero = generate_pairs_for_train_eval(gold_map_train, split='train',  previous_window =k)
        #print("final pairs train", len(train_pairs), len(train_labels), len(causal_probing_label_train))
        dev_pairs, dev_labels, causal_probing_label_dev, zero = generate_pairs_for_train_eval(gold_map_dev, split = 'dev',  previous_window =k)
        #print("final pairs dev", len(dev_pairs), len(dev_labels), len(causal_probing_label_dev)) 
        test_pairs, test_labels, causal_probing_label_test, zero = generate_pairs_for_train_eval(gold_map_test, split = 'test',  previous_window =k)
        #print("final pairs test", len(test_pairs), len(test_labels), len(causal_probing_label_test)) 
        
        print("pairs after duplicate removal", len(train_pairs), len(train_labels),len (dev_pairs), len(dev_labels), len(test_pairs), len(test_labels)) # 2948 training positive pairs
        print("positive samples in train", sum([x for x in train_labels if x ==1]))
        print("neg samples in train", len(train_labels) -  sum([x for x in train_labels if x ==1]))
        print("positive samples in dev", sum([x for x in dev_labels if x ==1]))
        print("neg samples in dev", len(dev_labels) - sum([x for x in dev_labels if x ==1]))  
        
            
        print("positive samples in test", sum([x for x in test_labels if x ==1]))
        print("neg samples in test", len(test_labels) -sum([x for x in test_labels if x ==1]))
        print("POS", sum(x for x in train_labels if x ==1), "NEG",  len(train_labels)-sum(x for x in train_labels if x ==1))
        print("POS", sum(x for x in dev_labels if x ==1),"NEG", len(dev_labels)-sum(x for x in dev_labels if x ==1))
        print("POS", sum(x for x in test_labels if x ==1),"NEG", len(test_labels)-sum(x for x in test_labels if x ==1))
