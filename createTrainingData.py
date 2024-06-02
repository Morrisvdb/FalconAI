import concurrent.futures
import os, json, random, re, csv, emoji, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from spellchecker import SpellChecker
from bs4 import BeautifulSoup
from langdetect import detect
nltk.download('stopwords')
nltk.download('wordnet')


BANNED_PHRASES = ["pluh"]
BLOCK_EMBEDS = True
USER = "nikevschalwisky"
OUTPUTNAME = "./trainingData.csv"
VAL_OUTPUTNAME = "./validationData.csv"
OUTPUT_PATH = "./data"
USED_FILES = [
                "data/The Crack Dynasty/general.json",
                "data/The Crack Dynasty/the-bois.json",
              ]
TOT_MESSAGES = None
VAL_LENGTH = 250
BATCH_SIZE = 5000

def txt_to_json(txt_file_path, json_file_path):
    print(f"Converting {txt_file_path} to {json_file_path}")
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()
        print(len(lines))
        messages = []
        for line in lines:
            message = {"author": line.split(':')[0], "content": line.split(':')[-1]}
            messages.append(message)
        
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(messages, json_file, indent=4)

input_type = input("Choose input type (Text=t/JSON=j): ")
if input_type not in ['t', 'j']:
    print("Invalid input type, aborting...")
    exit()
if input_type == None:
    input_type = 'j'

if input_type == 't':
    for data in os.listdir("data"):
        if data.endswith(".txt"):
            txt_to_json(f"data/{data}", f"data/temp.json")
            USED_FILES = [f"data/temp.json"]

def clean_message(message):
    # Remove HTML tags
    # message = BeautifulSoup(message, "html.parser").get_text()

    # Remove URLs
    message = re.sub(r'http\S+|www\S+|https\S+', '', message, flags=re.MULTILINE)

    # Remove emojis
    message = emoji.demojize(message)
    message = re.sub(r':[a-z_&]+:', '', message)

    # Remove non-alphanumeric characters
    # message = re.sub(r'\W+', ' ', message)

    # Convert to lowercase
    message = message.lower()

    return message

def validate_message(message):
    if message['content'] == '' or message['content'] == None:
        return False
    if re.match(r'^.*:.*:.*$', message['content']) is not None:
        return False
    if re.match(r'^.*<.*@.*>.*$', message['content']) is not None:
        return False
    for phrase in BANNED_PHRASES:
        if phrase in message['content']:
            return False
    # if BLOCK_EMBEDS and message['embeds'] != []:
    #     return False
    # if USER is not None and message['author'] != USER:
    #     return False
    if len(list(message['content'])) < 10:
        return False
    return True

import csv
def format_message(file_path):
    messages = []
    with open(file_path, 'r', encoding='utf-8') as input_file:
        # data = json.load(input_file)[:-VAL_LENGTH]
        data = json.load(input_file)
        for message in data[:-VAL_LENGTH]:
            if not validate_message(message):
                continue
            author = "BOT:" if message['author'] == USER else "USER:"
            # author = message['author']
            # if author != USER and USER is not None:
            #     continue
            content = clean_message(message['content'])
            messages.append([author, content])
            # messages.append(["MESSAGE: ", content])
    print(messages)
    return messages

def format_messages(file_paths, output_path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(format_message, file_path): file_path for file_path in file_paths}
        with open(os.path.join(OUTPUT_PATH, output_path), 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file)
            for future in concurrent.futures.as_completed(future_to_file):
                messages = future.result()
                writer.writerows(messages)

format_messages(USED_FILES, OUTPUTNAME)

with open(os.path.join(OUTPUT_PATH, OUTPUTNAME), 'r', encoding='utf-8') as input_file:
    data = list(csv.reader(input_file))
    TOT_MESSAGES = len(data)
    print(f"Total messages: {TOT_MESSAGES}")

def create_validation_data():
    for file_path in USED_FILES:
        messages = []
        with open(file_path, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)
            total_messages = 0
            for message in data[len(data) - VAL_LENGTH:]:
                author = message['author']
                content = message['content']
                messages.append([author, content])
                total_messages += 1
                if total_messages >= VAL_LENGTH:
                    break
    
    with open(os.path.join(OUTPUT_PATH, VAL_OUTPUTNAME), 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(messages)

create_validation_data()
