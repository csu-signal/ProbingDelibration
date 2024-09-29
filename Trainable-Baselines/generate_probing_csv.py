import pandas as pd
from WTD_GPT import WTDProcessor


weights_task_file_path = './Data/updated_weightsTaskDataset_withAug.csv'
weights_data = pd.read_csv(weights_task_file_path)

processor = WTDProcessor(weights_data)

probingQuestionsMap = processor.generate_probing_questions_map()
causal_promts = processor.generate_causal_prompts(probingQuestionsMap, processor=processor)
augmentation_prompts = processor.generate_probing_prompts(probingQuestionsMap)

responsesList = []

output_data = []
for key, value in probingQuestionsMap.items():
    row = {
        'key': key,
        'group_id': value['group_id'],
        'message_id': value['message_id'],
        'origin': value['origin'],
        'original_text': value['original_text'],
        'annotation_type': value['annotation_type'],
        'prev_utterance_history': value['prev_utterance_history'],
        'aug1': value['Augmentations'].get('aug1', ''),
        'aug2': value['Augmentations'].get('aug2', ''),
        'aug3': value['Augmentations'].get('aug3', ''),
        'causal_1': value['causal_utterances'].get('causal_1', ''),
        'causal_1_msgid': value['causal_utterances'].get('causal_1_msgid', ''),
        'causal_2': value['causal_utterances'].get('causal_2', ''),
        'causal_2_msgid': value['causal_utterances'].get('causal_2_msgid', ''),
        'causal_3': value['causal_utterances'].get('causal_3', ''),
        'causal_3_msgid': value['causal_utterances'].get('causal_3_msgid', '')
    }
    output_data.append(row)

output_df = pd.DataFrame(output_data)
output_df.to_csv('./Data/probing_questions_map.csv', index=False)
print("Data has been saved to probing_questions_map.csv")