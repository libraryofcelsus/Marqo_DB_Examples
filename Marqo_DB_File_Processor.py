import os
import json
import sys
import time
from datetime import datetime
from uuid import uuid4
import requests
import shutil
import importlib
from marqo import Client as MarqoClient
from marqo.errors import MarqoWebError
import traceback
import asyncio
from PyPDF2 import PdfReader
from ebooklib import epub
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from queue import Queue
import base64
import re
import gc

Debug_Output = "True"

def create_upload_folders():
    subfolders = [
        './Uploads/TXT',
        './Uploads/TXT/Finished',
        './Uploads/PDF',
        './Uploads/PDF/Finished',
        './Uploads/EPUB',
        './Uploads/EPUB/Finished',
        './Uploads/VIDEOS',
        './Uploads/VIDEOS/Finished',
        './Uploads/SCANS/Finished'
    ]

    for folder in subfolders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")


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
        response = requests.get("http://localhost:6333/dashboard/")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def initialize_marqo_client():
    with open('./Settings.json', 'r') as file:
        settings = json.load(file)
    marqo_url = settings.get('Marqo_URL', 'http://localhost:8882')
    client = MarqoClient(url=marqo_url)
    return client

mq = initialize_marqo_client()

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

def find_base64_encoded_json(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    pattern = re.compile(b'[A-Za-z0-9+/]{100,}={0,2}')
    matches = pattern.findall(binary_data)
    valid_json_objects = []
    for match in matches:
        try:
            decoded_data = base64.b64decode(match)
            json_data = json.loads(decoded_data)
            valid_json_objects.append(json_data)
        except (base64.binascii.Error, json.JSONDecodeError):
            continue
    return valid_json_objects

def load_format_settings(backend_model):
    file_path = f'./Model_Formats/{backend_model}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            formats = json.load(file)
    else:
        formats = {
            "user_input_start": "", 
            "user_input_end": "", 
            "assistant_input_start": "", 
            "assistant_input_end": "",
            "system_input_start": "", 
            "system_input_end": ""
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

async def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    end = chunk_size
    while end <= len(text):
        chunks.append(text[start:end])
        start += chunk_size - overlap
        end += chunk_size - overlap
    if end > len(text):
        chunks.append(text[start:])
    return chunks

async def Text_Extract():
    create_upload_folders()
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    bot_name = settings.get('Bot_Name', '')
    username = settings.get('Username', '')
    user_id = settings.get('User_ID', '')
    print("Enter a knowledge domain to assign to the files. To have the LLM assign one per chunk, type: 'Auto'")
    Domain = input().strip()  
    if not os.path.exists('./Uploads/TXT'):
        os.makedirs('./Uploads/TXT')
    if not os.path.exists('./Uploads/TXT/Finished'):
        os.makedirs('./Uploads/TXT/Finished')
    if not os.path.exists('./Uploads/PDF'):
        os.makedirs('./Uploads/PDF')
    if not os.path.exists('./Uploads/PDF/Finished'):
        os.makedirs('./Uploads/PDF/Finished')
    if not os.path.exists('./Uploads/EPUB'):
        os.makedirs('./Uploads/EPUB')
    if not os.path.exists('./Uploads/VIDEOS'):
        os.makedirs('./Uploads/VIDEOS')
    if not os.path.exists('./Uploads/VIDEOS/Finished'):
        os.makedirs('./Uploads/VIDEOS/Finished')
    if not os.path.exists('./Uploads/EPUB/Finished'):
        os.makedirs('./Uploads/EPUB/Finished')
    if not os.path.exists('./Uploads/SCANS/Finished'):
        os.makedirs('./Uploads/SCANS/Finished')
    
    semaphore = asyncio.Semaphore(1)  # Process one file at a time per host
    host_queue = Queue()
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        host_data = settings.get('HOST_AetherNode', '').strip()
        hosts = host_data.split(' ')
    except Exception as e:
        print(f"An error occurred while reading the host file: {e}")
    
    for host in hosts:
        host_queue.put(host)
    
    while True:
        try:
            timestamp = time.time()
            timestring = timestamp_to_datetime(timestamp)
            await process_files_in_directory('./Uploads/SCANS', './Uploads/SCANS/Finished', Domain, semaphore, host_queue)
            await process_files_in_directory('./Uploads/TXT', './Uploads/TXT/Finished', Domain, semaphore, host_queue)
            await process_files_in_directory('./Uploads/PDF', './Uploads/PDF/Finished', Domain, semaphore, host_queue)
            await process_files_in_directory('./Uploads/EPUB', './Uploads/EPUB/Finished', Domain, semaphore, host_queue)
            await process_files_in_directory('./Uploads/VIDEOS', './Uploads/VIDEOS/Finished', Domain, semaphore, host_queue)
            gc.collect() 
        except:
            traceback.print_exc()

async def process_files_in_directory(directory_path, finished_directory_path, Domain, semaphore, host_queue, chunk_size=700, overlap=80):
    try:
        files = os.listdir(directory_path)
        files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
        for file in files:
            await process_and_move_file(directory_path, finished_directory_path, file, Domain, semaphore, host_queue, chunk_size, overlap)
        gc.collect()  
    except Exception as e:
        print(e)
        traceback.print_exc()

async def process_and_move_file(directory_path, finished_directory_path, file, Domain, semaphore, host_queue, chunk_size, overlap):
    async with semaphore:
        try:
            file_path = os.path.join(directory_path, file)
            await chunk_text_from_file(file_path, Domain, host_queue, chunk_size, overlap)
            finished_file_path = os.path.join(finished_directory_path, file)
            shutil.move(file_path, finished_file_path)
        except Exception as e:
            print(e)
            traceback.print_exc()

async def chunk_text_from_file(file_path, Domain, host_queue, chunk_size=600, overlap=50):
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    bot_name = settings.get('Bot_Name', '')
    username = settings.get('Username', '')
    user_id = settings.get('User_ID', '')
    API = settings.get('API', 'AetherNode')
    Web_Search = settings.get('Search_Web', 'False')
    backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
    LLM_Model = settings.get('LLM_Model', 'AetherNode')
    Write_Dataset = settings.get('Write_To_Dataset', 'False')
    Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Custom')
    Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
    LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
    try:
        print("Reading given file, please wait...")
        pytesseract.pytesseract.tesseract_cmd = '.\\Tesseract-ocr\\tesseract.exe'
        texttemp = None
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.txt':
            with open(file_path, 'r') as file:
                texttemp = file.read().replace('\n', ' ').replace('\r', '')
        elif file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                texttemp = " ".join(page.extract_text() for page in pdf.pages)
        elif file_extension == '.epub':
            book = epub.read_epub(file_path)
            texts = []
            for item in book.get_items_of_type(9):  # type 9 is XHTML
                soup = BeautifulSoup(item.content, 'html.parser')
                texts.append(soup.get_text())
            texttemp = ' '.join(texts)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            image = Image.open(file_path)
            if image is not None:
                texttemp = pytesseract.image_to_string(image).replace('\n', ' ').replace('\r', '')
        elif file_extension in ['.mp4', '.mkv', '.flv', '.avi']:
            audio_file = "audio_extracted.wav"
            subprocess.run(["ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "44100", "-f", "wav", audio_file])

            model_stt = whisper.load_model("tiny")
            transcribe_result = model_stt.transcribe(audio_file)
            if isinstance(transcribe_result, dict) and 'text' in transcribe_result:
                texttemp = transcribe_result['text']
            else:
                print("Unexpected transcribe result")
                texttemp = ""  
            os.remove(audio_file) 
        else:
            print(f"Unsupported file type: {file_extension}")
            return []

        texttemp = '\n'.join(line for line in texttemp.splitlines() if line.strip())
        chunks = await chunk_text(texttemp, chunk_size, overlap)
        filelist = []
        collection_name = f"BOT_NAME_{bot_name}"
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
                    return

        try:
            for chunk in chunks:
                host = host_queue.get()
                await summarized_chunk_from_file(host, chunk, collection_name, bot_name, username, mq, file_path, Domain)
                host_queue.put(host)
        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred while processing chunks: {e}")

        return
    except Exception as e:
        print(e)
        traceback.print_exc()
        return "Error"

async def summarized_chunk_from_file(host, chunk, collection_name, bot_name, username, client, file_path, Domain):
    try:
        botnameupper = bot_name.upper()
        filelist = [] 
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        API = settings.get('API', 'AetherNode')
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
        filesum = list()
        filesum.append({
            'role': 'system',
            'content': """
            You are an AI assistant specializing in condensing articles while retaining all key information. Your task is to read the provided text and continue the conversation by presenting a concise version that:

            1. Keeps all essential information, including names, dates, numbers, and specific details.
            2. Uses natural language to present the information in a conversational manner.
            3. Maintains the original meaning and context.
            4. Shortens the text where possible without losing any content.
            5. Does not add any new information or interpretations.

            If no article is provided, simply respond with "I don't have any article to work with. What would you like to discuss?"

            Remember: Your goal is to present all the information from the original text in a more concise, conversational way.
            """
        })
        filesum.append({'role': 'user', 'content': f"Here's an article I'd like you to condense: {chunk}"})
        filesum.append({'role': 'assistant', 'content': "I've read the article you provided. I will now provide a concise version that retains all the key information.\nHere is a condensed version of the text: "})
        text_to_remove = f"SUMMARIZED ARTICLE: Based on the scraped text, here is the summary: "
        heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
        user_id = settings.get('User_ID', 'USER_ID')
        conv_length = settings.get('Conversation_Length', '3')
        Web_Search = settings.get('Search_Web', 'False')
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        LLM_Model = settings.get('LLM_Model', 'AetherNode')

        Write_Dataset = settings.get('Write_To_Dataset', 'False')
        Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Custom')
        Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
        prompt = ''.join([message_dict['content'] for message_dict in filesum])
        text = await Final_Response_API_Call(API, backend_model, filesum, username, bot_name)
        if text.startswith(text_to_remove):
            text = text[len(text_to_remove):].strip()
        if len(text) < 20:
            text = "No File available."

        fileyescheck = 'yes'
        if 'no file' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no article' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no summary' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'provide the article' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'i am an ai' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no article provided' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no file available' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        else:
            if 'cannot provide a summary of' in text.lower():
                text = chunk
            if 'yes' in fileyescheck.lower():
                semanticterm = list()
                semanticterm.append({'role': 'system', 'content': "MAIN SYSTEM PROMPT: You are a bot responsible for tagging articles with a question-based title for database queries. Your task is to read the provided text and generate a concise title in the form of a question that accurately represents the article's content. The title should be semantically identical to the article's overview, without including any extraneous information. Use the format: [<QUESTION TITLE>]."})

                semanticterm.append({'role': 'assistant', 'content': f"GIVEN ARTICLE: {text}"})

                semanticterm.append({'role': 'user', 'content': "Create a brief, single question that encapsulates the semantic meaning of the article. Use the format: [<QUESTION TITLE>]. Please only provide the question title, as it will be directly appended to the article."})

                semanticterm.append({'role': 'assistant', 'content': "ASSISTANT: Sure! Here's the semantic question tag for the article: "})

                text_to_remove = f"ASSISTANT: Sure! Here's the semantic question tag for the article: "

                prompt = ''.join([message_dict['content'] for message_dict in semanticterm])

                semantic_db_term = await Final_Response_API_Call(API, backend_model, semanticterm, username, bot_name)
                if semantic_db_term.startswith(text_to_remove):
                    semantic_db_term = semantic_db_term[len(text_to_remove):].strip()
                filename = os.path.basename(file_path)
                if 'cannot provide a summary of' in semantic_db_term.lower():
                    semantic_db_term = 'Tag Censored by Model'
                filelist.append(filename + ' ' + text)

                text_file_path = './Uploads/' + filename
                with open(text_file_path, 'a', encoding='utf-8') as f:
                    f.write('<' + filename + '>\n')
                    f.write('<' + semantic_db_term + '>\n')
                    f.write('<' + text + '>\n\n')

                if Domain == "Auto":
                    domain_extraction = []
                    domain_extraction = [
                        {'role': 'system', 'content': "You are a knowledge domain extractor. Your task is to identify the single, most general knowledge domain that best represents the given text. Respond with only one word for the domain, without any explanation or specifics."},
                        {'role': 'user', 'content': f"Text to analyze: {semantic_db_term} - {text}"},
                        {'role': 'assistant', 'content': "The most relevant knowledge domain for the given text is: "}
                    ]
                    text_to_remove = f"DOMAIN EXTRACTOR: The most relevant knowledge domain for the given text is: "
                    text_to_remove2 = f"DOMAIN EXTRACTOR:"
                    extracted_domain = await Final_Response_API_Call(API, backend_model, domain_extraction, username, bot_name)
                    extracted_domain = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, extracted_domain)
                    if extracted_domain.startswith(text_to_remove):
                        extracted_domain = extracted_domain[len(text_to_remove):].strip()
                    if extracted_domain.startswith(text_to_remove2):
                        extracted_domain = extracted_domain[len(text_to_remove2):].strip()
                    Domain = extracted_domain
                print('---------')
                print(filename + '\n' + semantic_db_term)
                print(f"\nEXTRACTED DOMAIN: {extracted_domain}\n\n")
                print(f"\n{text}")
                payload = list()
                timestamp = time.time()
                timestring = timestamp_to_datetime(timestamp)
                unique_id = str(uuid4())
                point_id = unique_id + str(int(timestamp))
                document = {
                    "_id": point_id,
                    "bot": bot_name,
                    "time": timestamp,
                    "message": text,
                    "timestring": timestring,
                    "uuid": unique_id,
                    "user": user_id,
                    "domain": Domain.upper(),
                    "memory_type": "External_Resources",
                    "document": text 
                }
                client.index(collection_name).add_documents(
                    [document],
                    tensor_fields=["document"], 
                    client_batch_size=1  
                )
                payload.clear()
                pass
            else:
                print('---------')
                print(f'\n\n\nERROR MESSAGE FROM BOT: {fileyescheck}\n\n\n')
        table = filelist
        return table
    except Exception as e:
        print(e)
        traceback.print_exc()
        table = "Error"
        return table

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
            "memory_type": "External_Resources",
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
    history = []
    while True:
        response = await Text_Extract()

if __name__ == "__main__":
    asyncio.run(main())
