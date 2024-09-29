
import re
import json
import openai
from delitoolkit.delidata import DeliData 
import time
import tiktoken
import pickle
from tqdm import tqdm
openai.api_key = '' # insert your secret api key 

class DeliDataProcessor:
    from delitoolkit.delidata import DeliData  # Assuming this import inside to keep the example concise

    def __init__(self, with_participant_info = True):
        self.delidata_corpus = DeliData()
        self.probing_count = {}
        self.probing_count_prev = {}
        self.message_history = {}  # Store all messages up to each probing question
        self.with_participant_info = with_participant_info
        
    def clean_prompt(self, text):
        # Strip leading and trailing whitespace
        text = text.strip()
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Optionally, replace newlines correctly if needed
        # text = text.replace('\n', ' ').strip()
        return text

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def get_probing_key(self, group_id):
        if group_id not in self.probing_count:
            self.probing_count[group_id] = 0
        self.probing_count[group_id] += 1
        return f"{group_id}_{self.probing_count[group_id]}"
    
    def get_probing_key_prev(self, group_id):
        if group_id not in self.probing_count_prev:
            self.probing_count_prev[group_id] = 0
        self.probing_count_prev[group_id] += 1
        return f"{group_id}_{self.probing_count_prev[group_id]}"

    def extract_prev_history(self):
        nested_dict = {}
        for group_id, messages in self.delidata_corpus.corpus.items():
            self.message_history[group_id] = []  # Initialize history for each group
            for message in messages:
                # Append message to history before checking if it is a probing question
                self.message_history[group_id].append(message)
                if message['annotation_type'] == 'Probing': #add participants information later
                    key = self.get_probing_key(group_id)
                    nested_dict[key] = message  # Store the whole message dict for simplicity
        return nested_dict
    
    def generate_probing_questions_map(self, prev_history_map, processor):
        probing_questions_map = {}
        for group_id, messages in self.delidata_corpus.corpus.items():
            for message in messages:
                if message['annotation_type'] == 'Probing':
                    key = self.get_probing_key_prev(group_id)
                    probing_questions_map[key] = {
                        'group_id': message['group_id'],
                        'message_id': message['message_id'],
                        'message_type': message['message_type'],
                        'origin': message['origin'],
                        'original_text': message['original_text'],
                        'clean_text': message['clean_text'],
                        'annotation_type': message['annotation_type'],
                        'annotation_target': message['annotation_target'],
                        'annotation_additional': message['annotation_additional'],
                        'team_performance': message['team_performance'],
                        'performance_change': message['performance_change'],
                        'sol_tracker_message': message['sol_tracker_message'],
                        'sol_tracker_all': message['sol_tracker_all']
#                         'prev_uttterance_history': prev_history_map[key]
                    }
        for key, value in probing_questions_map.items():
            group_id = value['group_id']
            message_id = value['message_id']
            previous_messages = processor.get_previous_messages(group_id, message_id)
            previous_messages = previous_messages.replace("\n", "")
            probing_questions_map[key]['prev_uttterance_history'] = previous_messages
        return probing_questions_map
    

    def get_previous_messages(self, group_id, message_id):
        """Get all messages up to but not including the message with the specified id."""
        history = self.message_history[group_id]
        previous_messages = []
        for message in history:
            if message['message_id'] == message_id:
                break
            if self.with_participant_info:
                
                previous_messages.append(message['origin'] + ":"+  message['original_text'])
            else:
                previous_messages.append(message['original_text'])
        return '\n'.join(previous_messages[1:len(previous_messages)]) #exclude the first message that contains participant information 

    def generate_prompts(self, nested_dict, processor):
        prompts = []
        for key, value in nested_dict.items():
            system_prompt = f'''
            You are a concise and expert annotator that follows 
            these instructions to extract 3 casual counterparts of
            a probing (reasoning) question (marked with #) in a collaborative task. 
            The causal counterparts are NOT same as the probing question but likely to be found close the the probing question in the dialogue history.
            Conversation Background: Participants are presented with 4 cards with a number or a letter on them.
            They have to answer the following question
            “Which cards should you turn to test the rule:
            All cards with vowels on one side have an even number on the other?”.
            HINT: The correct answer is to turn the vowel, to check for an even number 
            on the other side, and to turn the odd number, 
            to verify there isn't a vowel on the other side.
            '''
            system_prompt = "".join(system_prompt)

            system_prompt = system_prompt.replace("\n", "")
            # Fetch previous messages up to this probing question
            group_id = value['group_id']
            message_id = value['message_id']
            previous_messages = processor.get_previous_messages(group_id, message_id)
            if self.with_participant_info:
                 
                probing_question = "#" + value['origin'] + ":" + value['original_text'].replace("\n", "") + "#"
            else:
                
                probing_question = "#" + value['original_text'].replace("\n", "") + "#"
            previous_messages = previous_messages.replace("\n", "")
            data_dict = {
                'causal_counterpart_1': 'Your Answer',
                'causal_counterpart_2': 'Your Answer',
                'causal_counterpart_3':'Your Answer',
                'context': 'Your Answer',
                'reasoning_1': 'Your Answer',
                'reasoning_2': 'Your Answer',
                'reasoning_3': 'Your Answer'
            }
            formatted_json = json.dumps(data_dict, indent=4, sort_keys=False)
            prompt_text = f"{formatted_json}"
            
            
            user_prompt = f''' Use this context with participant information. 
            Probing Question: {probing_question} 
            Previous dialogue history:{previous_messages}
            Think step-by-step:
            1. Review the conversation (participant:utterance) and extract the utterance that directly triggered the probing question. This is the first causal counterpart.
            2. Identify the second and third causal counterparts (utterances) if they exist by tracing back to earlier relevant utterances.
            3. Provide a structured response in json format detailing the first, second, and third causal counterparts, including context and reasoning for each.
            Output format:{prompt_text}
        
            ''' 

            user_prompt = "".join(user_prompt)
            user_prompt = user_prompt.replace("\n", "")
             # Clean the prompts
            cleaned_system_prompt = self.clean_prompt(system_prompt)
            cleaned_user_prompt = self.clean_prompt(user_prompt)
            prompts.append({
                "system_prompt": cleaned_system_prompt,
                "user_prompt": cleaned_user_prompt
            })
        return prompts 

    
# Define the function to call GPT model with error handling, caching, and a delay before each call
    def call_gpt_with_prompt(self, messages,probing_map, model="gpt-3.5-turbo-0125", temperature=0.8, max_tokens=2000):
        cache_file = "gpt_responses.pkl" #cache the responses to reduce compute
        gpt_responses = {}

        # Load existing cache if available
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                gpt_responses = pickle.load(f)

        # Generate a unique key for the current prompt using serialization
        m_id = hash(json.dumps(messages, sort_keys=True))

        try:
            # Check if the response is already cached
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
            return response_evt, usage

        except openai.error.RateLimitError:
            print("Rate limit reached. Sleeping for 15 seconds...")
            time.sleep(15)
            return call_gpt_with_prompt(messages)
        except openai.error.APIError:
            print("API Error encountered. Sleeping for 15 seconds...")
            time.sleep(15)
            return call_gpt_with_prompt(messages)
        except openai.error.APIConnectionError:
            print("API Connection Error encountered. Sleeping for 15 seconds...")
            time.sleep(15)
            return call_gpt_with_prompt(messages)
        except openai.error.ServiceUnavailableError:
            print("Service Unavailable. Sleeping for 15 seconds...")
            time.sleep(15)
            return call_gpt_with_prompt(messages)

    def generate_gpt_rationales(self, prompts, full_probing_map):
        c = 0
        causal_counterpart_gpt_responses_file = 'causal_counterpart_gpt_delidata.pkl'
        prompts = prompts[58: 65]
        generated_causal_counterpart_map = {}  # Initialize the map
        with tqdm(total=len(prompts)) as pbar:
            for prompt, probing_key in zip(prompts, full_probing_map.keys()):
                try:
                    messages = [
                        {"role": "system", "content": prompt['system_prompt']},
                        {"role": "user", "content": prompt['user_prompt'] }
                    ]
                    response, usage = self.call_gpt_with_prompt(messages, full_probing_map)
                    print("usage", usage)
                    response = [json.loads(response)]
                    
                    causal_counterparts.append(response)
    
                    if self.with_participant_info: 
                        print("true for participant info")
                        probing_utterance = full_probing_map[probing_key]['origin'] + ":" + full_probing_map[probing_key]['original_text'] 
                        probing_utterance_masked = full_probing_map[probing_key]['origin'] + ":" + full_probing_map[probing_key]['clean_text']
                    else:
                        probing_utterance = full_probing_map[probing_key]['original_text'] 
                        probing_utterance_masked = full_probing_map[probing_key]['clean_text']
                    
                    generated_causal_counterpart_map[probing_key] = {
                        'causal_counterpart_1': response[0]['causal_counterpart_1'],
                        'causal_counterpart_2': response[0]['causal_counterpart_2'],
                        'causal_counterpart_3': response[0]['causal_counterpart_3'],
                        'reasoning_1': response[0]['reasoning_1'],
                        'reasoning_2': response[0]['reasoning_2'],
                        'reasoning_3': response[0]['reasoning_3'],
                        'context': response[0]['context'],
                        'probing_utterance': probing_utterance,
                        'probing_utterance_masked': probing_utterance_masked,
                        'prev_uttterance_history': full_probing_map[probing_key]['prev_uttterance_history'],
                        'usage_log': usage
                    }
                    pbar.update(1)
          
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue  # Continue to the next iteration of the loop
                
        # Dump the generated map to a file
        with open(causal_counterpart_gpt_responses_file, "wb") as f:
            pickle.dump(generated_causal_counterpart_map, f)
            
        return generated_causal_counterpart_map

def main(with_participant_info):
    
    processor = DeliDataProcessor(with_participant_info)
    prev_utterances_map = processor.extract_prev_history()
    #print("previous historykeys", prev_utterances_map.keys())
    full_probing_map = processor.generate_probing_questions_map(prev_utterances_map, processor)
    # #print("full_probing_map historykeys", full_probing_map.keys())
    prompts = processor.generate_prompts(prev_utterances_map, processor)
    print(prompts[23])
    # c = 0
    # causal_counterparts = []
    # generated_causal_counterpart_map = {}
    # print("prompt len and probing map dimensions", len(prompts),len(full_probing_map.keys()))
    # generated_causal_counterpart_map = processor.generate_gpt_rationales(prompts,full_probing_map)

    # return generated_causal_counterpart_map
   

if __name__ == "__main__":
    generated_causal_counterpart_map = main(with_participant_info= True)



