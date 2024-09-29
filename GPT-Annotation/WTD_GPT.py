import pandas as pd
import os 
import time
import re
import json
import openai
from delitoolkit.delidata import DeliData 
import time
# import tiktoken
import pickle
from tqdm import tqdm
openai.api_key = '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' 
import WTD_GPT

class WTDProcessor:
    def __init__(self, data):
        self.data = data
        self.probing_count = {}
        self.message_history = {}  # Store all messages up to each probing question
        self.probing_count_prev = {}  # Dictionary to count consecutive probing questions per group
        
    def get_probing_key_prev(self, group_id):
        """Generate a unique key for consecutive probing questions in a group."""
        if group_id not in self.probing_count_prev:
            self.probing_count_prev[group_id] = 0
        self.probing_count_prev[group_id] += 1
        return f"{group_id}_{self.probing_count_prev[group_id]}"
    def clean_prompt(self, text):
        # Strip leading and trailing whitespace
        text = text.strip()
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Optionally, replace newlines correctly if needed
        # text = text.replace('\n', ' ').strip()
        return text


    #this method takes in a message id and gets the previous 22 messages
    def get_previous_messages(self, group_id, message_id):
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]
        # Filter data for the specified group and where the origin is not 4. 4 is Mariah
        group_data = self.data[(self.data['group_id'] == group_id) & (self.data['origin'] != 4)].reset_index(drop=True)
        if group_data.empty:
            print("No previous messages found.")
        # Find the index of the specified message_id in the filtered data
        current_index = group_data[group_data['message_id'] == message_id].index.min()
        #print('current index,' , current_index)
        
        # Calculate the starting index for previous messages, ensuring it doesn't go below zero
        start_index = max(0, current_index - 18)
        
        # Get previous messages from start_index up to but not including current_index
        previous_messages = group_data.iloc[start_index:current_index]
       
        # Inclide particiant and their text
        message_history = [
            f"Participant {row['origin']}: {row['original_text']}"
            for _, row in previous_messages.iterrows()
        ]
        #print(message_history)
        
        # Join the messages with new lines and return the result
        return "\n".join(message_history)
    
    def get_future_messages(self, group, current_index):
        group_data = self.data[self.data['group_id'] == group]
        end_index = min(len(group_data), current_index + 6)  # Include the current + 5 future messages
        future_messages = group_data.iloc[current_index+1:end_index]
        message_history = [
            f"Participant {row['origin']}: {row['original_text']}"
            for _, row in future_messages.iterrows()
        ]
        return "\n".join(message_history)

    def get_message_id_by_utterance(self, utterance):
        """
        Searches the dataset for the given utterance and returns the corresponding message ID.
        If no match is found, returns None.
        """
        # Remove the participant prefix and clean the utterance
        cleaned_utterance = utterance.split(': ', 1)[1] if ': ' in utterance else utterance
        # Search for the cleaned utterance in the dataset
        matches = self.data[self.data['original_text'].str.contains(cleaned_utterance, na=False, regex=False)]
        if not matches.empty:
            return matches['message_id'].iloc[0]
        else:
            return None 
    
    def generate_probing_questions_map(self):
        probing_questions_map = {}
        for _, row in self.data.iterrows():
            if row['annotation_type'] == 'Probing':
                key = self.get_probing_key_prev(row['group_id'])
                previous_messages = self.get_previous_messages(row['group_id'], row['message_id'])
                
                # Extract and process augmentations
                # augmentations = {
                #     'aug1': row['Augmented_Probing_1'],
                #     'aug2': row['Augmented_Probing_2'],
                #     'aug3': row['Augmented_Probing_3']
                # }

                # Prepare and extract causal counterparts, removing the participant prefix
                # causal_utterances = {}
                # for i in range(1, 4):
                #     counterpart_key = f'causal_counterpart_{i}'
                #     if pd.notna(row[counterpart_key]):
                #         cleaned_utterance = row[counterpart_key].split(': ', 1)[1] if ': ' in row[counterpart_key] else row[counterpart_key]
                #         # Search for the cleaned utterance in the dataset to find its message ID
                #         message_id = self.get_message_id_by_utterance(row[counterpart_key])
                #         causal_utterances[f'causal_{i}'] = cleaned_utterance
                #         causal_utterances[f'causal_{i}_msgid'] = message_id

                probing_questions_map[key] = {
                    'group_id': row['group_id'],
                    'message_id': row['message_id'],
                    'origin': row['origin'],
                    'original_text': row['original_text'],
                    'annotation_type': row['annotation_type'],
                    'prev_utterance_history': previous_messages
                    # 'Augmentations': augmentations,
                    # 'causal_utterances': causal_utterances
                }
                    
        return probing_questions_map

    import json

    def generate_causal_prompts(self, nested_dict, processor):
        prompts = []
        for key, value in nested_dict.items():
            # Adjust the description according to the context of the WTD task
            system_prompt = '''
            You are a concise and expert annotator tasked with identifying causal counterparts 
            of a probing (reasoning) question in a collaborative task. Causal counterparts are 
            not the same as the probing question but are likely found close to the probing question 
            in the dialogue history. 
            Conversation Background: Participants are first given a balance scale to determine the weights of five colorful wooden blocks. "
            "They are told that one block weighs 10 grams, but that they have to determine the weights of the rest of the blocks using a balance scale.
            
            Extract up to three causal counterparts.
            '''

            # Simplify and clean the prompt
            system_prompt = system_prompt.replace("\n", " ").strip()

            # Fetch previous messages up to this probing question
            group_id = value['group_id']
            message_id = value['message_id']
            previous_messages = processor.get_previous_messages(group_id, message_id)
            previous_messages = previous_messages.replace("\n", " ")

            # Format the probing question
            probing_question = "#" + value['original_text'].replace("\n", "") + "#"

            # Placeholder JSON structure
            data_dict = {
                'causal_counterpart_1': 'Your Answer',
                'causal_counterpart_2': 'Your Answer',
                'causal_counterpart_3': 'Your Answer',
                'context': 'your Answer',
                'reasoning_1': 'Your Answer',
                'reasoning_2': 'Your Answer',
                'reasoning_3': 'Your Answer'
            }
            formatted_json = json.dumps(data_dict, indent=4, sort_keys=False)
            prompt_text = f"{formatted_json}"
            # Create user prompt
            user_prompt = f'''Use the following context and identify causal counterparts. 
            Probing Question: {probing_question} 
            Previous dialogue history: {previous_messages}
            Think step-by-step:
            1. Review the conversation (participant:utterance) and extract the utterance that directly triggered the probing question. This is the first causal counterpart.
            2. Identify the second and third causal counterparts (utterances) if they exist by tracing back to earlier relevant utterances.
            3. Provide a structured response in json format detailing the first, second, and third causal counterparts, including context and reasoning for each.
            Output format:{prompt_text}

            '''

            # Clean the prompts
            cleaned_system_prompt = self.clean_prompt(system_prompt)
            cleaned_user_prompt = self.clean_prompt(user_prompt)

            # Append to the list of prompts
            prompts.append({
                "system_prompt": cleaned_system_prompt,
                "user_prompt": cleaned_user_prompt,
                "message_id": message_id
            })

        return prompts
    def annotate_probing_prompt(self):
        prompts = []
        participantData = self.data[self.data['origin'] != 4] 
        for idx, row in participantData.iterrows():
            system_prompt = (
                "You are a concise and expert annotator. Label the following utterance as 'probing', 'non-probing deliberation' , or 'Neither' "
                "based on the following definitions:\n"
                "1. Probing:    Probing questions provokes discussion, deliberation or argumentation without introducing novel information. Such utterances could be considered conversational interventions that may change the flow of the "
                "conversation to induce further arguments or to moderate a conversation \n"
                "2. Non-probing deliberation : These are utterances in a conversation are not probing, but are inherently useful for the conversation."
                "they include all discussions that are concerned with the task's solution and participants' reasoning"
                "3. Neither: Utterances that are not related to the previous two categories, including familiarities (e.g., 'Greetings fellas') or hesitation cues (e.g., 'hmm...').\n\n"
                "Task Description: Participants are first given a balance scale to determine the weights of five colorful wooden blocks. "
                "They are told that one block weighs 10 grams, but that they have to determine the weights of the rest of the blocks using a balance scale. "
                "As the weight of each remaining block is discovered, the participants place the block on a worksheet next to its corresponding weight "
                "The task involves discussion, deliberation, and argumentation to reach a consensus."
                
            )
            future_messages = self.get_future_messages(row['group_id'], idx)
            
            data_dict = {
                'label': 'Your label',
                'rationale': 'Your rationale',
               
            }
            formatted_json = json.dumps(data_dict, indent=4, sort_keys=False)
            prompt_text = f"{formatted_json}"
            
            user_prompt = (
                #f"Future dialogue context:\n{future_messages}\n\n"
                f"Current Utterance:\n{row['original_text']}\n\n"
                "Provide a structured response in JSON format detailing rationale."
                " Provide a structured response in json format"
                f" Output format:{prompt_text} "
            )

            prompts.append({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "index": idx
            })
            
        return prompts
    def generate_probing_prompts(self, nested_dict):
        probing_prompts = []
        for key, value in nested_dict.items():
            # System with context
            system_prompt = f"""
            Participants are first given a balance scale to determine the weights of five colorful wooden blocks. "
            "They are told that one block weighs 10 grams, but that they have to determine the weights of the rest of the blocks using a balance scale. "
            "As the weight of each remaining block is discovered, the participants place the block on a worksheet next to its corresponding weight. 
            A 'probing' utterance in this context is a question or statement that prompts further discussion or consideration without 
            introducing new information. It challenges the current reasoning or asks for justification of a decision.

            Context for the original probing question:
          
            Original Probing Utterance: {value['original_text']}
            Dialogue History: {self.get_previous_messages(value['group_id'], value['message_id'])}
            """
            data_dict = {
                'Augmented Probing 1': 'Your Answer',
                'Augmented Probing 2': 'Your Answer',
                'Augmented Probing 3': 'Your Answer'
               
            }
            formatted_json = json.dumps(data_dict, indent=4, sort_keys=False)
            prompt_text = f"{formatted_json}"
            # User prompt with instructions
            user_prompt = f"""
            Generate three new probing utterances that are semantically similar but syntactically different, based on the dialogue history and the nature of the original probing question.
            " Provide a structured response in json format"
            " Output format:{prompt_text} "
        
            """

            #Clean 
            cleaned_system_prompt = self.clean_prompt(system_prompt)
            cleaned_user_prompt = self.clean_prompt(user_prompt)

            #For GPT call
            probing_prompts.append({
                "system_prompt": cleaned_system_prompt,
                "user_prompt": cleaned_user_prompt,
                "message_id": value['message_id']
            })
        
        return probing_prompts

    
    def call_gpt_with_prompt(self, messages, model="gpt-3.5-turbo-0125", temperature=0.8, max_tokens=2000, pickleName = 'gpt_responses_annotations'):
        cache_file = f'{pickleName}.pkl'#cache the responses to reduce compute
        gpt_responses = {}

        # Load existing cache if available
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                gpt_responses = pickle.load(f)

        # Generate a unique key for the current prompt using serialization
        m_id = hash(json.dumps(messages, sort_keys=True))

        try:
            # Check if the response is already cached
            messages = [
                        {"role": "system", "content": messages['system_prompt']},
                        {"role": "user", "content": messages['user_prompt'] }
                    ]
            if m_id not in gpt_responses:
                print("Fetching new response from GPT...")
                # Delay before each API call
                time.sleep(1)
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages
                )
                response_evt = response.choices[0].message['content']
                usage = response["usage"]
                gpt_responses[m_id] = {"prompt": messages, "response": response_evt}

                # Save the updated cache
                with open(cache_file, "wb") as f:
                    pickle.dump(gpt_responses, f)
            else:
                print("Using cached response...")
                response_evt = gpt_responses[m_id]["response"]

            # Return the response
            return response_evt

        except openai.error.RateLimitError:
            print("Rate limit reached. Sleeping for 15 seconds...")
            time.sleep(15)
            return self.call_gpt_with_prompt(messages)
        except openai.error.APIError:
            print("API Error encountered. Sleeping for 15 seconds...")
            time.sleep(15)
            return self.call_gpt_with_prompt(messages)
        except openai.error.APIConnectionError:
            print("API Connection Error encountered. Sleeping for 15 seconds...")
            time.sleep(15)
            return self.call_gpt_with_prompt(messages)
        except openai.error.ServiceUnavailableError:
            print("Service Unavailable. Sleeping for 15 seconds...")
            time.sleep(15)
            return self.call_gpt_with_prompt(messages)    
        




weights_task_file_path = 'wtd_with_probing.csv'
weights_data = pd.read_csv(weights_task_file_path)
#Create an instance of the WTDProcessor class
processor = WTDProcessor(weights_data)
# print(processor.get_previous_messages(group_id=1, message_id='msgid31'))
# print(processor.generate_probing_questions_map())
probingQuestionsMap = processor.generate_probing_questions_map()
failed_causals = ['msgid297', 'msgid524', 'msgid1297', 'msgid1618', 'msgid1943']

causal_promts = processor.generate_causal_prompts(probingQuestionsMap, processor=processor)
print(len(causal_promts))
for i in range(len(causal_promts)):
    message_id = causal_promts[i]['message_id']
    if(message_id not in failed_causals):
            continue
    response = processor.call_gpt_with_prompt(messages=causal_promts[i], pickleName='gpt_probe_augmentation_final')

    try:
        response = json.loads(response)
       # print(response)

        # Fetch the message_id from the prompt that corresponds to this response
        
        # Find the index in the DataFrame using the message_id
        index = weights_data[weights_data['message_id'] == message_id].index

        # Update the DataFrame with the response data for causal counterparts and reasoning
        weights_data.loc[index, 'causal_counterpart_1'] = response.get('causal_counterpart_1', 'NA')
        weights_data.loc[index, 'causal_counterpart_2'] = response.get('causal_counterpart_2', 'NA')
        weights_data.loc[index, 'causal_counterpart_3'] = response.get('causal_counterpart_3', 'NA')

        weights_data.loc[index, 'reasoning_1'] = response.get('reasoning_1', 'NA')
        weights_data.loc[index, 'reasoning_2'] = response.get('reasoning_2', 'NA')
        weights_data.loc[index, 'reasoning_3'] = response.get('reasoning_3', 'NA')
    #   'causal_counterpart_1': 'Your Answer',
    #             'causal_counterpart_2': 'Your Answer',
    #             'causal_counterpart_3': 'Your Answer',
    #             'context': 'your Answer',
    #             'reasoning_1': 'Your Answer',
    #             'reasoning_2': 'Your Answer',
    #             'reasoning_3': 'Your Answer'
        
        # Save the DataFrame to CSV
        weights_data.to_csv('wtd_with_probing_and_causal_18.csv', index=False)

    except json.JSONDecodeError:
        print(f"Failed to decode JSON for response: {response}")
        failed_causals.append(message_id)
        continue  # Skip this entry and proceed with the next iteration
    except TypeError:
        print(f"Unexpected type error with response: {response}")
        continue 

#     print(causal_promts[i])
#     response = processor.call_gpt_with_prompt(messages=causal_promts[i], pickleName='gpt_causal')
#     responsesList.append(response)
# augmentation_prompts = processor.generate_probing_prompts(probingQuestionsMap)
# print(probingQuestionsMap)
# responsesList = []
print(len(failed_causals))
print(failed_causals)
# print(len(augmentation_prompts))
# for i in range(len(augmentation_prompts)):
#     response = processor.call_gpt_with_prompt(messages=augmentation_prompts[i], pickleName='gpt_probe_augmentation_final')

#     try:
#         response = json.loads(response)
#         print(response)

#         # Fetch the message_id from the prompt that corresponds to this response
#         message_id = augmentation_prompts[i]['message_id']

#         # Find the index in the DataFrame using the message_id
#         index = weights_data[weights_data['message_id'] == message_id].index

#         # Update the DataFrame with the response data for causal counterparts and reasoning
#         weights_data.loc[index, 'Augmented_Probing_1'] = response.get('Augmented Probing 1', 'NA')
#         weights_data.loc[index, 'Augmented_Probing_2'] = response.get('Augmented Probing 2', 'NA')
#         weights_data.loc[index, 'Augmented_Probing_3'] = response.get('Augmented Probing 3', 'NA')
      
        
#         # Save the DataFrame to CSV
#         weights_data.to_csv('updated_weightsTaskDataset_withAug.csv', index=False)

#     except json.JSONDecodeError:
#         print(f"Failed to decode JSON for response: {response}")
#         continue  # Skip this entry and proceed with the next iteration
#     except TypeError:
#         print(f"Unexpected type error with response: {response}")
#         continue 

# #     print(causal_promts[i])
# #     response = processor.call_gpt_with_prompt(messages=causal_promts[i], pickleName='gpt_causal')
# #     responsesList.append(response)
# # print(response)
# # Generate prompts

#UNCOMMENT THE REST FOR PROBING ANNOTATION. COST: $.57
# prompts = processor.annotate_probing_prompt()

# for i in range(len(prompts)):
#     # if(i > 5):
#     #     break
#     response = processor.call_gpt_with_prompt(messages=prompts[i])
#     try:
        
#         response = json.loads(response)
#         print(response)
#         weights_data.at[prompts[i]['index'], 'annotation_type'] = response['label']
#         weights_data.at[prompts[i]['index'], 'rationale_for_probing_label'] = response['rationale']
#         weights_data.to_csv('updated_weightsTaskDataset.csv', index=False)
#     except json.JSONDecodeError:
#         print(f"Failed to decode JSON for response: {response}")
#         continue  # Skip this entry and proceed with the next iteration
#     except TypeError:
#         print(f"Unexpected type error with response: {response}")
#         continue 
# #Save the updated dataset to a new CSV file
# updated_file_path = 'wtd_with_probing.csv'
# weights_data.to_csv(updated_file_path, index=False)
