import pickle
import torch
from helper_probing import tokenize, tokenize_utterances, forward_ab, f1_score, accuracy, precision, recall
from probing_prediction import predict_causal_counterpart
import random
from tqdm import tqdm
import os
from models import CrossEncoder
from delitoolkit.delidata import DeliData
import pickle
import json 
import pandas as pd

def generate_split_pairs_labels(dataset, causal_counterpart_map, document_map):

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
    print("bad indices", len(pos_bad_indices), len(neg_bad_indices))
    print(pos_bad_indices)
   
    positive_pairs_cleaned = [(x,y) for index, (x,y) in enumerate(positive_pairs) if index not in pos_bad_indices]
    negative_pairs_cleaned = [(x,y) for index, (x,y) in enumerate(negative_pairs) if index not in neg_bad_indices]
    print(len(positive_pairs_cleaned), len(negative_pairs_cleaned))
    #print(positive_pairs_cleaned)
    print(positive_pairs_cleaned)
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
    return train_pairs, train_labels, dev_pairs, dev_labels, test_pairs, test_labels


def combine_non_probing_with_probing_map(dataset):
    
    # causal_counterpart_gpt_responses_file = f'{dataset}/final_probing_cc_map.pkl'
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
    wtd = True
    if(not wtd):
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
    print(generated_causal_counterpart_map['1_1'])

    return generated_causal_counterpart_map, document_map


def train_dpos(dataset, model_name=None):
    dataset = 'deli_data'
    dataset = 'wtd_dataset'
    dataset_folder = f'./datasets/{dataset}/'
       

    device = torch.device('cuda:1')
    device_ids = list(range(1))
    device_ids = [1]
#     train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_oracle/mp_mp_t_train.pkl', 'rb'))
#     dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_oracle/mp_mp_t_dev.pkl', 'rb'))

#     tps_train, fps_train, _, _ = train_mp_mpt
#     tps_dev, fps_dev, _, _ = dev_mp_mpt

#     train_pairs = list(tps_train + fps_train)
#     train_labels = [1]*len(tps_train) + [0]*len(fps_train)

#     dev_pairs = list(tps_dev + fps_dev)
#     dev_labels = [1] * len(tps_dev) + [0] * len(fps_dev)
    
    
    #get the maps 
    probing_map, document_map = combine_non_probing_with_probing_map(dataset = 'wtd_dataset')
    
    #generate the train and dev sets here 
    
    train_pairs, train_labels, dev_pairs, dev_labels, test_pairs, test_labels = generate_split_pairs_labels(dataset, probing_map, document_map)
    print(len(train_pairs), len(train_labels),len (dev_pairs), len(dev_labels), len(test_pairs), len(test_labels)) # 2948 training positive pairs 
    
    #try a sample unit test with a smaller pos/neg set
    # train_pairs =train_pairs[0:100] + train_pairs[-100:] # 50 pos and 50 neg labels
    # train_labels = train_labels[0:100] + train_labels[-100:] 
    # dev_pairs = dev_pairs[0:50] + dev_pairs[-50:]  # 50 pos and 50 neg labels
    # dev_labels = dev_labels[0:50] + dev_labels[-50:]
    # print(" smaller prototype", len(train_pairs), len(train_labels),len (dev_pairs), len(dev_labels), len(test_pairs), len(test_labels)) # 2948 training positive pairs
    

    # model_name = 'roberta-base'
    scorer_module = CrossEncoder(is_training=True, long = True, model_name=model_name).to(device)

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    print("special tokens", parallel_model.module.start_id, parallel_model.module.end_id) 
     
    
    
    train_ab,train_ba, dev_ab, dev_ba = train(train_pairs, train_labels,dev_pairs, dev_labels, test_pairs, test_labels, parallel_model, probing_map,document_map, dataset_folder, device,
          batch_size=4, n_iters=10, lr_lm=0.000001, lr_class=0.0001) 
    return train_ab,train_ba, dev_ab, dev_ba, parallel_model

def train(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          test_pairs, 
          test_labels,
          parallel_model,
          probing_map,
          document_map,
          working_folder,
          device,
          batch_size=4,
          n_iters=50,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_easy_hard_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer
    print('Train pairs ', train_pairs)
    # prepare data
    train_ab, train_ba = tokenize_utterances(tokenizer, train_pairs, probing_map, document_map, parallel_model.module.end_id,  max_sentence_len=512)
    dev_ab, dev_ba = tokenize_utterances(tokenizer, dev_pairs, probing_map, document_map, parallel_model.module.end_id, max_sentence_len=512)
    test_ab, test_ba = tokenize_utterances(tokenizer, test_pairs, probing_map, document_map, parallel_model.module.end_id, max_sentence_len=512)

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    test_labels = torch.LongTensor(test_labels)
    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        # new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices)

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

            scores_mean = (scores_ab + scores_ba) / 2

            loss = bce_loss(scores_mean, batch_labels)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_scores_ab, dev_scores_ba = predict_causal_counterpart(parallel_model, dev_ab, dev_ba, device, batch_size)
        dev_predictions = (dev_scores_ab + dev_scores_ba)/2
        dev_predictions = dev_predictions > 0.5
        dev_predictions = torch.squeeze(dev_predictions)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev recall:", recall(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))

        test_scores_ab, test_scores_ba = predict_causal_counterpart(parallel_model, test_ab, test_ba, device, batch_size)
        test_predictions = (test_scores_ab + test_scores_ba)/2
        test_predictions = test_predictions > 0.5
        test_predictions = torch.squeeze(test_predictions)

        print("test accuracy:", accuracy(test_predictions, test_labels))
        print("test precision:", precision(test_predictions, test_labels))
        print("test recall:", recall(test_predictions, test_labels))
        print("test f1:", f1_score(test_predictions, test_labels))
# #         if n % 2 == 0:
# #             scorer_folder = working_folder + f'/scorer/chk_{n}'
# #             if not os.path.exists(scorer_folder):
# #                 os.makedirs(scorer_folder)
# #             model_path = scorer_folder + '/linear.chkpt'
# #             torch.save(parallel_model.module.linear.state_dict(), model_path)
# #             parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
# #             parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
# #             print(f'saved model at {n}')

# #     scorer_folder = working_folder + '/scorer/'
# #     if not os.path.exists(scorer_folder):
# #         os.makedirs(scorer_folder)
# #     model_path = scorer_folder + '/linear.chkpt'
# #     torch.save(parallel_model.module.linear.state_dict(), model_path)
# #     parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
# #     parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

    return train_ab,train_ba, dev_ab, dev_ba 
if __name__ == '__main__':
    
    train_ab,train_ba, dev_ab, dev_ba, parallel_model = train_dpos('ecb', model_name='allenai/longformer-base-4096')
