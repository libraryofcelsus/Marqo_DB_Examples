import sys
import os
import json
import time
from datetime import datetime
from uuid import uuid4
import requests
import importlib
import numpy as np
import re
import traceback
import asyncio
import aiofiles
import aiohttp
import base64
import marqo
from marqo.errors import MarqoWebError

# Set up global variables
Debug_Output = "True"
Memory_Output = "False"
Dataset_Output = "False"

embed_size = "768"

def embeddings(query):
    vector = model.encode([query])[0].tolist()
    return vector

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
       return file.read().strip()

def timestamp_func():
    try:
        return time.time()
    except:
        return time()

def check_local_server_running():
    try:
        response = requests.get("http://localhost:8882")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

if check_local_server_running():
    mq = marqo.Client(url="http://localhost:8882")
else:
    try:
        with open('./Settings.json', 'r') as file:
            settings = json.load(file)
        api_key = settings.get('Marqo_API_Key', '')
        mq = marqo.Client(url="https://api.marqo.ai", api_key=api_key)
    except:
        print("\n\nMarqo is not started. Please enter API Keys or run Marqo Locally.")
        sys.exit()

def is_url(string):
    return string.startswith('http://') or string.startswith('https://')

def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str

def import_api_function():
    settings_path = './Settings.json'
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    api_module_name = settings['API']
    module_path = f'./Resources/API_Calls/{api_module_name}.py'
    spec = importlib.util.spec_from_file_location(api_module_name, module_path)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    llm_api_call = getattr(api_module, 'LLM_API_Call', None)
    input_expansion_api_call = getattr(api_module, 'Input_Expansion_API_Call', None)
    inner_monologue_api_call = getattr(api_module, 'Inner_Monologue_API_Call', None)
    intuition_api_call = getattr(api_module, 'Intuition_API_Call', None)
    final_response_api_call = getattr(api_module, 'Final_Response_API_Call', None)
    short_term_memory_response_api_call = getattr(api_module, 'Short_Term_Memory_API_Call', None)
    if llm_api_call is None:
        raise ImportError(f"LLM_API_Call function not found in {api_module_name}.py")
    return llm_api_call, input_expansion_api_call, inner_monologue_api_call, intuition_api_call, final_response_api_call, short_term_memory_response_api_call

def load_format_settings(backend_model):
    file_path = f'./Model_Formats/{backend_model}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            formats = json.load(file)
    else:
        formats = {
            "heuristic_input_start": "",
            "heuristic_input_end": "",
            "system_input_start": "",
            "system_input_end": "",
            "user_input_start": "", 
            "user_input_end": "", 
            "assistant_input_start": "", 
            "assistant_input_end": ""
        }
    return formats

def set_format_variables(backend_model):
    format_settings = load_format_settings(backend_model)
    heuristic_input_start = format_settings.get("heuristic_input_start", "")
    heuristic_input_end = format_settings.get("heuristic_input_end", "")
    system_input_start = format_settings.get("system_input_start", "")
    system_input_end = format_settings.get("system_input_end", "")
    user_input_start = format_settings.get("user_input_start", "")
    user_input_end = format_settings.get("user_input_end", "")
    assistant_input_start = format_settings.get("assistant_input_start", "")
    assistant_input_end = format_settings.get("assistant_input_end", "")
    return heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end

def find_base64_encoded_json(binary_data):
    pattern = re.compile(b'(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?')
    matches = pattern.findall(binary_data)
    valid_json_objects = []
    for match in matches:
        if len(match) % 4 != 0:
            continue
        try:
            decoded_data = base64.b64decode(match, validate=True)
            decoded_str = decoded_data.decode('utf-8')
            json_data = json.loads(decoded_str)
            if isinstance(json_data, dict) and 'spec' in json_data:
                valid_json_objects.append(json_data)
        except (base64.binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return valid_json_objects

def format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, response):
    try:
        if response is None:
            return "ERROR WITH API"  
        if backend_model == "Llama_3":
            assistant_input_start = "assistant"
            assistant_input_end = "assistant"
        botname_check = f"{botnameupper}:"
        while (response.startswith(assistant_input_start) or response.startswith('\n') or
               response.startswith(' ') or response.startswith(botname_check)):
            if response.startswith(assistant_input_start):
                response = response[len(assistant_input_start):]
            elif response.startswith(botname_check):
                response = response[len(botname_check):]
            elif response.startswith('\n'):
                response = response[1:]
            elif response.startswith(' '):
                response = response[1:]
            response = response.strip()
        botname_check = f"{botnameupper}: "
        if response.startswith(botname_check):
            response = response[len(botname_check):].strip()
        if backend_model == "Llama_3":
            if "assistant\n" in response:
                index = response.find("assistant\n")
                response = response[:index]
        if response.endswith(assistant_input_end):
            response = response[:-len(assistant_input_end)].strip()
        
        return response
    except:
        traceback.print_exc()
        return ""  

def write_dataset_custom(backend_model, heuristic, system, user_input, output):
    data = {
        "heuristic": heuristic,
        "system": system,
        "input": user_input,
        "output": output
    }
    try:
        with open(f'{backend_model}_custom_dataset.json', 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(f'{backend_model}_custom_dataset.json', 'w') as file:
            json.dump([data], file, indent=4)

def write_dataset_simple(backend_model, user_input, output):
    data = {
        "input": user_input,
        "output": output
    }

    try:
        with open(f'{backend_model}_simple_dataset.json', 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(f'{backend_model}_simple_dataset.json', 'w') as file:
            json.dump([data], file, indent=4)

class MainConversation:
    def __init__(self, username, user_id, bot_name, max_entries):
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        self.format_config = self.initialize_format(backend_model)
        
        self.bot_name_upper = bot_name.upper()
        self.username_upper = username.upper()
        self.max_entries = int(max_entries)
        self.file_path = f'./History/{user_id}/{bot_name}_Conversation_History.json'
        self.main_conversation = [] 

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.running_conversation = data.get('running_conversation', [])
        else:
            self.running_conversation = []
            self.save_to_file()

    def initialize_format(self, backend_model):
        file_path = f'./Model_Formats/{backend_model}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                formats = json.load(file)
        else:
            formats = {
                "user_input_start": "", 
                "user_input_end": "", 
                "assistant_input_start": "", 
                "assistant_input_end": ""
            }
        return formats

    def format_entry(self, user_input, response, initial=False):
        user = f"{self.username_upper}: {user_input}"
        bot = f"{self.bot_name_upper}: {response}"
        return {'user': user, 'bot': bot}

    def append(self, timestring, user_input, response):
        entry = self.format_entry(f"[{timestring}] - {user_input}", response)
        self.running_conversation.append(entry)
        while len(self.running_conversation) > self.max_entries:
            self.running_conversation.pop(0)
        self.save_to_file()

    def save_to_file(self):
        data_to_save = {
            'main_conversation': self.main_conversation,
            'running_conversation': self.running_conversation
        }
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def get_conversation_history(self):
        formatted_history = []
        for entry in self.running_conversation: 
            user_entry = entry['user']
            bot_entry = entry['bot']
            formatted_history.append(user_entry)
            formatted_history.append(bot_entry)
        return '\n'.join(formatted_history)
        
    def get_dict_conversation_history(self):
        formatted_history = []
        for entry in self.running_conversation:
            user_entry = {'role': 'system', 'content': entry['user']}
            bot_entry = {'role': 'assistant', 'content': entry['bot']}
            formatted_history.append(user_entry)
            formatted_history.append(bot_entry)
        return formatted_history

    def get_dict_formated_conversation_history(self, user_input_start, user_input_end, assistant_input_start, assistant_input_end):
        formatted_history = []
        for entry in self.running_conversation:
            user_entry = {'role': 'user', 'content': f"{user_input_start}{entry['user']}{user_input_end}"}
            bot_entry = {'role': 'assistant', 'content': f"{assistant_input_start}{entry['bot']}{assistant_input_end}"}
        return formatted_history

    def get_last_entry(self):
        if self.running_conversation:
            return self.running_conversation[-1]
        return None
    
    def delete_conversation_history(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            self.running_conversation = []
            self.save_to_file()

def load_character_file(file_name):
    file_path = f'Characters/{file_name}.png'
    if not os.path.exists(file_path):
        print(f"Character file not found: {file_path}")
        return None
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    json_objects = find_base64_encoded_json(binary_data)
    return json_objects[0] if json_objects else None

async def load_prompts(user_id, bot_name):
    base_path = "./Chatbot_Prompts"
    base_prompts_path = os.path.join(base_path, "Base")
    user_bot_path = os.path.join(base_path, user_id, bot_name)  
    if not os.path.exists(user_bot_path):
        os.makedirs(user_bot_path)
    prompts_json_path = os.path.join(user_bot_path, "prompts.json")
    base_prompts_json_path = os.path.join(base_prompts_path, "prompts.json")
    if not os.path.exists(prompts_json_path) and os.path.exists(base_prompts_json_path):
        async with aiofiles.open(base_prompts_json_path, 'r') as base_file:
            base_prompts_content = await base_file.read()
        async with aiofiles.open(prompts_json_path, 'w') as user_file:
            await user_file.write(base_prompts_content)
    async with aiofiles.open(prompts_json_path, 'r') as file:
        return json.loads(await file.read())

def ensure_marqo_index_exists(collection_name):
    try:
        mq.index(collection_name).get_stats()
    except MarqoWebError as e:
        if e.code == 'index_not_found':
            try:
                mq.create_index(collection_name)
                print(f"Index {collection_name} created successfully.")
            except Exception as create_error:
                print(f"Error creating index: {str(create_error)}")
                sys.exit(1)
        else:
            raise e

async def Marqo_DB_Chatbot_Long_Term_Memory(user_input, username, user_id, bot_name, main_conversation, character_data, prompts, LLM_API_Call, Input_Expansion_API_Call, collection_name, image_path=None):
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    API = settings.get('API', 'AetherNode')
    conv_length = settings.get('Conversation_Length', '3')
    Web_Search = settings.get('Search_Web', 'False')
    backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
    LLM_Model = settings.get('LLM_Model', 'AetherNode')
    Use_Char_Card = settings.get('Use_Character_Card', 'False')
    Char_File_Name = settings.get('Character_Card_File_Name', 'Aetherius')
    Write_Dataset = settings.get('Write_To_Dataset', 'False')
    Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Custom')
    Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
    LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
    input_expansion = list()
    domain_extraction = list()
    conversation = list()
    heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
    end_prompt = ""
    base_path = "./Chatbot_Prompts"
    base_prompts_path = os.path.join(base_path, "Base")
    user_bot_path = os.path.join(base_path, user_id, bot_name)  
    if not os.path.exists(user_bot_path):
        os.makedirs(user_bot_path)
    prompts_json_path = os.path.join(user_bot_path, "prompts.json")
    base_prompts_json_path = os.path.join(base_prompts_path, "prompts.json")
    if not os.path.exists(prompts_json_path) and os.path.exists(base_prompts_json_path):
        async with aiofiles.open(base_prompts_json_path, 'r') as base_file:
            base_prompts_content = await base_file.read()
        async with aiofiles.open(prompts_json_path, 'w') as user_file:
            await user_file.write(base_prompts_content)
    async with aiofiles.open(prompts_json_path, 'r') as file:
        prompts = json.loads(await file.read())
    main_prompt = prompts["main_prompt"].replace('<<NAME>>', bot_name)
    greeting_msg = prompts["greeting_prompt"].replace('<<NAME>>', bot_name)
    if Use_Char_Card == 'True' and character_data:
        json_data = character_data
        bot_name = json_data['data']['name']
    else:
        Char_File_Name = bot_name
    botnameupper = bot_name.upper()
    usernameupper = username.upper()
    collection_name = f"BOT_NAME_{Char_File_Name}"
    main_conversation = MainConversation(username, user_id, bot_name, conv_length)
    while True:
        try:
            conversation_history = main_conversation.get_dict_conversation_history()
            con_hist = main_conversation.get_conversation_history()
            timestamp = timestamp_func()
            timestring = timestamp_to_datetime(timestamp)

            if Use_Char_Card == 'True' and character_data:
                json_data = character_data
                bot_name = json_data['data']['name']
                greeting_msg = json_data['data']['first_mes']
                system_prompt = json_data['data']['system_prompt']
                personality = json_data['data']['personality']
                description = json_data['data']['description']  
                scenario = json_data['data']['scenario']
                example_format = json_data['data']['mes_example']
                greeting_msg = greeting_msg.replace("{{user}}", username).replace("{{char}}", bot_name)
                system_prompt = system_prompt.replace("{{user}}", username).replace("{{char}}", bot_name)
                personality = personality.replace("{{user}}", username).replace("{{char}}", bot_name)
                description = description.replace("{{user}}", username).replace("{{char}}", bot_name)
                scenario = scenario.replace("{{user}}", username).replace("{{char}}", bot_name)
                example_format = example_format.replace("{{user}}", username).replace("{{char}}", bot_name)
                if len(example_format) > 3:
                    new_prompt = f"{system_prompt}\nUse the following format:{example_format}"
                else:
                    new_prompt = system_prompt
                main_prompt = f"{scenario}\n{personality}\n{description}"

                end_prompt = json_data['data']['post_history_instructions']
                character_tags = json_data['data']['tags']
                author_notes = json_data['data']['creator_notes']
            else:
                Char_File_Name = bot_name
            
            input_expansion.append({'role': 'system', 'content': f"You are a task rephraser. Your primary task is to rephrase the user's most recent input succinctly and accurately. Please return the rephrased version of the user’s most recent input. USER'S MOST RECENT INPUT: {user_input}"})
            input_expansion.append({'role': 'user', 'content': f"PREVIOUS CONVERSATION HISTORY: {con_hist}\n\n\nUSER'S MOST RECENT INPUT: {user_input}\n\n\n"})
            input_expansion.append({'role': 'assistant', 'content': f"TASK REPHRASER: Sure! Here's the rephrased version of the user's most recent input: "}) 
            if API == "OpenAi":
                expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
            if API == "Oobabooga":
                expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
            if API == "KoboldCpp":
                expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in input_expansion])
                expanded_input = await Input_Expansion_API_Call(API, prompt, username, bot_name)
            expanded_input = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, expanded_input)
            if Debug_Output == "True":
                print(f"\n\nEXPANDED INPUT: {expanded_input}\n\n")
                    
            
            def remove_duplicate_dicts(input_list):
                output_list = []
                for item in input_list:
                    if item not in output_list:
                        output_list.append(item)
                return output_list
            
            print("START DB SEARCH")
            external_search = ""
            try:
                # Vector search
                vector_results = mq.index(f"BOT_NAME_{Char_File_Name}").search(
                    q=expanded_input,
                    limit=15,
                    filter_string=f"user:{user_id} AND memory_type:Long_Term",
                    search_method="TENSOR"
                )
                vector_hits = vector_results['hits']

                # Lexical search
                lexical_results = mq.index(f"BOT_NAME_{Char_File_Name}").search(
                    q=expanded_input,
                    limit=15,
                    filter_string=f"user:{user_id} AND memory_type:Long_Term",
                    search_method="LEXICAL"
                )
                lexical_hits = lexical_results['hits']

                # Combine vector and lexical results
                combined_hits = vector_hits + lexical_hits

                # Remove duplicates
                unique_hits = remove_duplicate_dicts(combined_hits)

                # Sort by timestring
                sorted_hits = sorted(unique_hits, key=lambda x: x['timestring'])

                # Extract the message field
                external_search = [hit['message'] for hit in sorted_hits]

            except Exception as e:
                if "index not found" in str(e).lower():
                    external_search = ""
                else:
                    print(f"\nAn unexpected error occurred: {str(e)}")
                    external_search = ""

            if Debug_Output == "True":
                print(f"\nCombined search results:\n{external_search}\n")

            
            new_prompt = f"""
{main_prompt}

Your primary tasks are:
1. Respond to {username}'s messages using the provided memories and conversation history.
2. Prioritize information from memories over your built-in knowledge.
3. Maintain a natural, engaging conversation flow.

Specific instructions:

1. Memory usage:
   - Always check the provided memories first when answering questions or continuing conversations.
   - Only use your built-in knowledge if no relevant information is found in the memories.

2. Conversation style:
   - Respond in a friendly, conversational manner.
   - Adjust your tone to match {username}'s style and mood.
   - Use contractions and casual language to sound more natural (e.g., "I'm" instead of "I am").

3. Response format:
   - Keep responses concise and to the point.
   - Break longer responses into shorter paragraphs for readability.
   - Use emojis sparingly if appropriate for the conversation tone.

4. Continuing the conversation:
   - Reference previous parts of the conversation to maintain context.
   - Ask follow-up questions to show interest and keep the conversation flowing.
   - If {username} mentions a new topic, acknowledge it before responding.

5. Handling uncertainty:
   - If you're unsure about something, admit it politely.
   - Offer to find more information or suggest where {username} might look for answers.

Remember to always stay in character as {bot_name} and never break the fourth wall by mentioning that you're an AI language model.
""" 
            user_prompt_1 = f"Here is your past memories from previous conversations, use any information contained inside over latent knowledge.\nMEMORY WINDOW: [{external_search}]"
            
            assistant_prompt_1 = f"Thank you for providing the context window, please now provide the conversation with the user.\n<CONVERSATION START>"
            assistant_prompt_2 = f"Thank you for providing the context window, please now provide {username}'s current message."
            

            conversation = [
                {'role': 'system', 'content': f"{new_prompt}\n\n{main_prompt}"},
                {'role': 'user', 'content': f"Here are your past memories from previous conversations. Use this information over your built-in knowledge when possible:\n\nMEMORY: {external_search}"},
                {'role': 'assistant', 'content': f"I understand. I'll use the provided memories and continue our conversation naturally."}
            ]

            if len(conversation_history) > 0:
                conversation.append({'role': 'user', 'content': "Here's our conversation so far:"})
                if len(greeting_msg) > 1:
                    conversation.append({'role': 'assistant', 'content': f"{botnameupper}: {greeting_msg}"})
                for entry in conversation_history:
                    conversation.append(entry)

            conversation.append({'role': 'user', 'content': f"{usernameupper}: {user_input}"})
            conversation.append({'role': 'assistant', 'content': f"Please respond as {bot_name}. Use memories and information from previous conversations to guide your response.\n\n{botnameupper}: "})
            if API == "OpenAi":
                final_response = await LLM_API_Call(API, backend_model, conversation, username, bot_name)
            if API == "Oobabooga":
                final_response = await LLM_API_Call(API, backend_model, conversation, username, bot_name)
            if API == "KoboldCpp":
                final_response = await LLM_API_Call(API, backend_model, conversation, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in conversation])
                final_response = await LLM_API_Call(API, prompt, username, bot_name)
            conversation.clear()    
            final_response = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, final_response)
            print(f"\n\nFINAL RESPONSE: {final_response}\n\n")
                
                
            domain_extraction = []        
            domain_extraction = [
                {'role': 'system', 'content': 
                 """You are a Domain Ontology Specialist. Your task is to extract a single, general knowledge domain from the given text or query.

                 Instructions:
                 1. Analyze the provided text carefully.
                 2. Identify the main topic or theme.
                 3. Choose a single word that best represents the general knowledge domain of the text.
                 4. Respond with only this single word.
                 5. Do not include any explanations or comments.

                 Examples:
                 - For a text about photosynthesis in plants: "Biology"
                 - For a query about the American Civil War: "History"
                 - For a discussion on computer programming: "Technology"

                 Remember: Your response should be a single word only."""
                },
                {'role': 'user', 'content': f"Extract the domain from this text: {final_response}"}
            ]
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in domain_extraction])
                extracted_domain = await LLM_API_Call(prompt, username, bot_name)
            if API == "OpenAi":
                extracted_domain = LLM_API_Call(API, backend_model, domain_extraction, username, bot_name)
            if API == "KoboldCpp":
                extracted_domain = await LLM_API_Call(API, backend_model, domain_extraction, username, bot_name)
            if API == "Oobabooga":
                extracted_domain = await LLM_API_Call(API, backend_model, domain_extraction, username, bot_name)
            domain_extraction.clear()    
            extracted_domain = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, extracted_domain)
            if Debug_Output == "True":
                print(f"\n\nDOMAIN: {extracted_domain}\n\n")
            
            response_pair = f"{usernameupper}: {user_input}\n{botnameupper}: {final_response}"
                    
            success = upload_conversation_entry(bot_name, user_id, response_pair, extracted_domain, collection_name)
            if success:
                print("Conversation entry uploaded successfully.")
            else:
                print("Failed to upload conversation entry.")
            
            context_check = f"{external_search}"
            dataset = []
            llama_3 = "Llama_3"
            heuristic_input_start2, heuristic_input_end2, system_input_start2, system_input_end2, user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2 = set_format_variables(Dataset_Format)
            formated_conversation_history = main_conversation.get_dict_formated_conversation_history(user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2)

            if len(context_check) > 10:
                dataset_prompt_1 = f"Here is your context window for factual verification:\nCONTEXT WINDOW: [{external_search}]"
                dataset_prompt_2 = f"Thank you for providing the context window, please now provide the conversation with the user."
                dataset.append({'role': 'user', 'content': f"{user_input_start2}{dataset_prompt_1}{user_input_end2}"})
                dataset.append({'role': 'assistant', 'content': f"{assistant_input_start2}{dataset_prompt_2}{assistant_input_end2}"})
                dataset.append({'role': 'assistant', 'content': f"I will now provide the previous conversation history:"})
                
            if len(formated_conversation_history) > 1:
                if len(greeting_msg) > 1:
                    dataset.append({'role': 'assistant', 'content': f"{greeting_msg}"})
                for entry in formated_conversation_history:
                    dataset.append(entry)

            dataset.append({'role': 'user', 'content': f"{user_input_start2}{user_input}{user_input_end2}"})
            filtered_content = [entry['content'] for entry in dataset if entry['role'] in ['user', 'assistant']]
            llm_input = '\n'.join(filtered_content)
            heuristic = f"{heuristic_input_start2}{main_prompt}{heuristic_input_end2}"
            system_prompt = f"{system_input_start2}{new_prompt}{system_input_end2}"
            assistant_response = f"{assistant_input_start2}{final_response}{assistant_input_end2}"
            if Dataset_Output == 'True':
                print(f"\n\nHEURISTIC: {heuristic}")
                print(f"\n\nSYSTEM PROMPT: {system_prompt}")
                print(f"\n\nINPUT: {llm_input}")  
                print(f"\n\nRESPONSE: {assistant_response}")
                     

            if Write_Dataset == 'True':
                print(f"\n\nWould you like to write to dataset? Y or N?")   
                while True:
                    try:
                        yorno = input().strip().upper() 
                        if yorno == 'Y':
                            print(f"\n\nWould you like to include the conversation history? Y or N?")
                            while True:
                                yorno2 = input().strip().upper() 
                                if yorno2 == 'Y':
                                    if Dataset_Upload_Type == 'Custom':
                                        write_dataset_custom(Dataset_Format, heuristic, system_prompt, llm_input, assistant_response)
                                        print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                                    if Dataset_Upload_Type == 'Simple':
                                        write_dataset_simple(Dataset_Format, llm_input, final_response)
                                        print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
                                    break  
                                elif yorno2 == 'N':
                                    if Dataset_Upload_Type == 'Custom':
                                        write_dataset_custom(Dataset_Format, heuristic, system_prompt, user_input, assistant_response)
                                        print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                                    if Dataset_Upload_Type == 'Simple':
                                        write_dataset_simple(Dataset_Format, user_input, final_response)
                                        print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
                                    break 
                                else:
                                    print("Invalid input. Please enter 'Y' or 'N'.")

                            break  
                        elif yorno == 'N':
                            print("Not written to Dataset.\n\n")
                            break 
                        else:
                            print("Invalid input. Please enter 'Y' or 'N'.")
                    except:
                        traceback.print_exc()
            if Write_Dataset == 'Auto':
                if Dataset_Upload_Type == 'Custom':
                    write_dataset_custom(Dataset_Format, heuristic, system_prompt, llm_input, assistant_response)
                    print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                if Dataset_Upload_Type == 'Simple':
                    write_dataset_simple(Dataset_Format, user_input, final_response)
                    print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")

            main_conversation.append(timestring, user_input, final_response)
            if Debug_Output == 'True':
                print("\n\n\n")
            return heuristic, system_prompt, llm_input, user_input, final_response
        except:
            error = traceback.print_exc()
            error1 = traceback.print_exc()
            error2 = traceback.print_exc()
            error3 = traceback.print_exc()
            error4 = traceback.print_exc()
            return error, error1, error2, error3, error4

def upload_conversation_entry(bot_name, user_id, text, domain, collection_name):
    try:
        # Check if the index exists
        try:
            mq.index(collection_name).get_stats()
            print(f"Index {collection_name} already exists.")
        except MarqoWebError as e:
            if e.code == 'index_not_found':
                print(f"Index {collection_name} not found. Creating it now...")
                try:
                    mq.create_index(collection_name)
                    print(f"Index {collection_name} created successfully.")
                except Exception as create_error:
                    print(f"Error creating index: {str(create_error)}")
                    return False
            else:
                raise e

        timestamp = time.time()
        timestring = timestamp_to_datetime(timestamp)
        unique_id = str(uuid4())
        point_id = unique_id + str(int(timestamp))

        # Create the document with all metadata
        document = {
            "_id": point_id,
            "bot": bot_name,
            "time": timestamp,
            "message": text,
            "timestring": timestring,
            "uuid": unique_id,
            "user": user_id,
            "domain": domain.upper(),
            "memory_type": "Long_Term",
            "document": text  # The main content to be embedded
        }

        # Add the document to the Marqo index
        mq.index(collection_name).add_documents(
            [document],
            tensor_fields=["document"],  # Field to be embedded
            client_batch_size=1  # Process one document at a time
        )

        print(f"Successfully added document with ID: {point_id}")
        return True

    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        return False

async def main():
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    username = settings.get('Username', 'User')
    user_id = settings.get('User_ID', 'UNIQUE_USER_ID')
    bot_name = settings.get('Bot_Name', 'Chatbot')
    conv_length = settings.get('Conversation_Length', '3')
    Use_Char_Card = settings.get('Use_Character_Card', 'False')
    Char_File_Name = settings.get('Character_Card_File_Name', 'Aetherius')
    backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
    
    # Load character data once at the start
    character_data = load_character_file(Char_File_Name) if Use_Char_Card == "True" else None
    
    # Load prompts once at the start
    prompts = await load_prompts(user_id, bot_name)
    greeting_msg = prompts["greeting_prompt"].replace('<<NAME>>', bot_name)
    
    if Use_Char_Card == "True" and character_data:
        bot_name = character_data['data']['name'] 
        greeting_msg = character_data['data']['first_mes']
        greeting_msg = greeting_msg.replace("{{user}}", username).replace("{{char}}", bot_name)
    
    # Import API functions once at the start
    LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
    
    # Set format variables once at the start
    heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
    
    # Ensure Marqo index exists
    collection_name = f"BOT_NAME_{Char_File_Name}"
    ensure_marqo_index_exists(collection_name)
    
    # Create MainConversation object once
    
    
    history = []
    while True:
        main_conversation = MainConversation(username, user_id, bot_name, conv_length)
        conversation_history = main_conversation.get_dict_conversation_history()
        con_hist = main_conversation.get_conversation_history()
        print(con_hist)
        if len(conversation_history) < 1:
            print(f"{bot_name}: {greeting_msg}\n")
        user_input = input(f"{username}: ")
        if user_input.lower() == 'exit':
            break
        response = await Marqo_DB_Chatbot_Long_Term_Memory(user_input, username, user_id, bot_name, main_conversation, character_data, prompts, LLM_API_Call, Input_Expansion_API_Call, collection_name)
        history.append({"user": user_input, "bot": response})

if __name__ == "__main__":
    asyncio.run(main())
