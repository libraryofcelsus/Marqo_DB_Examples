import sys
import os
import json
import time
import datetime as dt
from datetime import datetime
from uuid import uuid4
import requests
import shutil
import importlib
import numpy as np
import re
import keyboard
import traceback
import asyncio
import aiofiles
import aiohttp
import base64
from marqo.errors import MarqoWebError
import sys
import os
import json
import time
import datetime as dt
from datetime import datetime
from uuid import uuid4
import requests
import shutil
import importlib
import numpy as np
import re
import keyboard
import traceback
import asyncio
import aiofiles
import aiohttp
import base64
import marqo


Debug_Output = "True"
Memory_Output = "False"
Dataset_Output = "False"


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
        with open(settings_path, 'r') as file:
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



async def Marqo_DB_Chatbot_RAG(user_input, username, user_id, bot_name, image_path=None):
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    API = settings.get('API', 'AetherNode')
    conv_length = settings.get('Conversation_Length', '3')
    backend_model = settings.get('Model_Backend', 'Llama_3')
    LLM_Model = settings.get('LLM_Model', 'Oobabooga')
    Write_Dataset = settings.get('Write_To_Dataset', 'False')
    Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Custom')
    Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
    LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
    input_expansion = list()
    domain_selection = list()
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
    botnameupper = bot_name.upper()
    usernameupper = username.upper()
    collection_name = f"BOT_NAME_{bot_name}"
    main_conversation = MainConversation(username, user_id, bot_name, conv_length)
    while True:
        try:
            conversation_history = main_conversation.get_dict_conversation_history()
            con_hist = main_conversation.get_conversation_history()
            timestamp = timestamp_func()
            timestring = timestamp_to_datetime(timestamp)
       #     input_2 = None
       #     if len(user_input) > 2:
       #         input_2 = user_input
       #     if image_path is not None:
       #         print(f" Sending: {image_path}  to Vision Model")
       #         loop = asyncio.get_event_loop()
       #         try:
       #             user_input = await loop.run_in_executor(None, gpt_vision, user_input, image_path)
       #             if input_2 is not None and len(input_2) > 2:
       #                 user_input = f"VISION: {user_input}\nORIGINAL USER INQUIRY: {input_2}"
       #         except:
       #             print(f"VISION MODEL FAILED")
       #             user_input = f"VISION MODEL FAILED.  INFORM USER TO CHECK OPENAI API KEY"


            input_expansion.append({'role': 'system', 'content': f"You are a task rephraser. Your job is to restate the user's most recent input clearly and concisely. Focus on capturing the main points and intent of the user's message."})
            input_expansion.append({'role': 'user', 'content': f"CONTEXT: {con_hist}\n\nLATEST USER INPUT: {user_input}\n\nPlease rephrase the latest user input, taking into account the context if relevant."})
            input_expansion.append({'role': 'assistant', 'content': f"Rephrased user input: "}) 
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

            domain_selection.append({
                'role': 'system',
                'content': """You are a Domain Ontology Specialist. Your task is to extract the single most important topic from the given text or user query. Follow these guidelines:

            1. Identify the main subject or topic of the text.
            2. Choose a single word that best represents this topic.
            3. The word should be a noun and as specific as possible while still being general enough to categorize the text.
            4. This word will be used to sort and categorize database queries related to the text.
            5. Respond ONLY with the chosen word. Do not include any explanations, comments, or punctuation.

            Example input: "What are the benefits of regular exercise for cardiovascular health?"
            Example output: Exercise

            Example input: "Can you explain the process of photosynthesis in plants?"
            Example output: Photosynthesis"""
            })
            domain_selection.append({
                'role': 'user', 'content': f"Extract the main topic from this text: {expanded_input}"})
            if API == "OpenAi":
                selected_domain = await Input_Expansion_API_Call(API, backend_model, domain_selection, username, bot_name)
            if API == "Oobabooga":
                selected_domain = await Input_Expansion_API_Call(API, backend_model, domain_selection, username, bot_name)
            if API == "KoboldCpp":
                selected_domain = await Input_Expansion_API_Call(API, backend_model, domain_selection, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in domain_selection])
                selected_domain = await Input_Expansion_API_Call(API, prompt, username, bot_name)
            selected_domain = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, selected_domain)
            if Debug_Output == "True":
                print(f"\n\nSELECTED DOMAIN: {selected_domain}\n\n")
                
                    
            
            def remove_duplicate_dicts(input_list):
                output_list = []
                for item in input_list:
                    if item not in output_list:
                        output_list.append(item)
                return output_list
            
            
            external_search = ""
            try:
                # Vector search
                vector_results = mq.index(f"BOT_NAME_{bot_name}").search(
                    q=expanded_input,
                    limit=18,
                    filter_string=f"user:{user_id} AND memory_type:External_Resources",
                    search_method="TENSOR"
                )
                vector_hits = [hit['message'] for hit in vector_results['hits']]

                # Lexical search
                lexical_results = mq.index(f"BOT_NAME_{bot_name}").search(
                    q=selected_domain,
                    limit=18,
                    filter_string=f"user:{user_id} AND memory_type:External_Resources",
                    search_method="LEXICAL"
                )
                lexical_hits = [hit['message'] for hit in lexical_results['hits']]
                
                # Combine Results
                combined_hits = vector_hits + lexical_hits
                external_search = remove_duplicate_dicts(combined_hits)
                external_search = "\n".join([f"[ - {entry}]" for entry in external_search])

            except Exception as e:
                if "index not found" in str(e).lower():
                    external_search = ""
                else:
                    print(f"\nAn unexpected error occurred: {str(e)}")
                    external_search = ""
            if Debug_Output == "True":
                print(f"\nCombined search results:\n{external_search}\n")
                
                
                
           # <task> Create DB Pruner
            pruner = []
            pruner.append({'role': 'system', 'content': 
                """You are a database entry pruner. Your task is to analyze individual database entries and remove those that are completely irrelevant to the user's question. Do not combine or mix information from different entries. Keep relevant entries intact and in their original format."""})

            pruner.append({'role': 'assistant', 'content': 
                f"DATABASE ENTRIES: {external_search}"})

            pruner.append({'role': 'user', 'content': 
                f"""USER QUESTION: {expanded_input}

                TASK:
                Prune the database entries based on relevance to the user's question. Follow these steps:

                1. Read the user's question carefully.
                2. Review each database entry individually.
                3. Keep entries that contain relevant information to the user's question.
                4. Remove entries that are completely irrelevant.
                5. Do not modify or combine entries - keep them in their original format.
                6. Include the [- Entry] markers for each kept entry.
                7. If unsure about relevance, keep the entry.

                OUTPUT:
                Provide only the pruned list of relevant entries, each in its original format. Do not add any explanations or ask any questions."""})

            pruner.append({'role': 'assistant', 'content': 
                "PRUNED DATABASE ENTRIES: "})
                
            if API == "OpenAi":
                pruned_entries = await LLM_API_Call(API, backend_model, pruner, username, bot_name)
            if API == "Oobabooga":
                pruned_entries = await LLM_API_Call(API, backend_model, pruner, username, bot_name)
            if API == "KoboldCpp":
                pruned_entries = await LLM_API_Call(API, backend_model, pruner, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in pruner])
                pruned_entries = await LLM_API_Call(API, prompt, username, bot_name) 
                
            if Debug_Output == "True":
                print(f"\nPruned Entries:\n{pruned_entries}\n")  
                
                
 
            new_prompt = f"You are {bot_name}.  Your task is to answer the user, {username}'s input using the provided context for factual verification.  If the needed information isn't contained in the provided context, only print: INFORMATION NOT FOUND IN DATABASE.  Only provide the user with an answer if the information is contained in the context window, do not use latent knowledge to answer any inquiry. Assume all data not provided by the context window is outdated and incorrect." 
            user_prompt_1 = f"Here is your context window for factual verification, use any information contained inside over latent knowledge.\nCONTEXT WINDOW: [{external_search}]"
            assistant_prompt_1 = f"Thank you for providing the context window, please now provide the conversation with the user."
            assistant_prompt_2 = f"Thank you for providing the context window, please now provide {username}'s current message."

            conversation.append({'role': 'system', 'content': f"{main_prompt}\n{new_prompt}"})
            conversation.append({'role': 'user', 'content': f"{user_prompt_1}"})

            if len(conversation_history) > 2:
                conversation.append({'role': 'assistant', 'content': f"{assistant_prompt_1}"})
                if len(greeting_msg) > 2:
                    conversation.append({'role': 'assistant', 'content': f"{greeting_msg}"})
                for entry in conversation_history:
                    conversation.append(entry)
            if len(end_prompt) > 2:
                conversation.append({'role': 'system', 'content': f"{end_prompt}"})
            conversation.append({'role': 'assistant', 'content': f"{assistant_prompt_2}"})    
            conversation.append({'role': 'user', 'content': f"CURRENT USER INQUIRY TO ANSWER: {user_input}"})
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
            
            context_check = f"{external_search}"
            dataset = []
            llama_3 = "Llama_3"
            heuristic_input_start2, heuristic_input_end2, system_input_start2, system_input_end2, user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2 = set_format_variables(Dataset_Format)
            formated_conversation_history = main_conversation.get_dict_formated_conversation_history(user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2)

            if len(context_check) > 10:
                dataset_prompt_1 = f"Here is your context window for factual verification, use any information contained inside over latent knowledge.\nCONTEXT WINDOW: [{external_search}]"
                dataset_prompt_2 = f"Thank you for providing the context window, please now provide the conversation with the user."
                dataset.append({'role': 'user', 'content': f"{user_input_start2}{dataset_prompt_1}{user_input_end2}"})
                dataset.append({'role': 'assistant', 'content': f"{assistant_input_start2}{dataset_prompt_2}{assistant_input_end2}"})
                dataset.append({'role': 'user', 'content': f"I will now provide the previous conversation history:"})
                
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
            


async def main():
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    username = settings.get('Username', 'User')
    user_id = settings.get('User_ID', 'UNIQUE_USER_ID')
    bot_name = settings.get('Bot_Name', 'Chatbot')
    conv_length = settings.get('Conversation_Length', '3')
    history = []
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
    greeting_msg = prompts["greeting_prompt"].replace('<<NAME>>', bot_name)
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
        response = await Marqo_DB_Chatbot_RAG(user_input, username, user_id, bot_name)
        history.append({"user": user_input, "bot": response})

if __name__ == "__main__":
    asyncio.run(main())
    
    
    
