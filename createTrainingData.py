import concurrent.futures
import os, json, random, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from bs4 import BeautifulSoup
import emoji
from langdetect import detect
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


BANNED_PHRASES = ["pluh"]
BLOCK_EMBEDS = True
USER = None
OUTPUTNAME = "//wsl.localhost/Ubuntu/home/user/Programming/AI/trainingData.csv"
VAL_OUTPUTNAME = "//wsl.localhost/Ubuntu/home/user/Programming/AI/validationData.txt"
USED_FILES = [
                "The Crack Dynasty/general.json",
                "The Crack Dynasty/the-bois.json",
              ]
TOT_MESSAGES = None
VAL_LENGTH = 2500
BATCH_SIZE = 5000

import concurrent.futures

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize the spell checker
spell = SpellChecker()

# Set of stopwords in English
stop_words = set(stopwords.words('english'))

def clean_message(message):
    # Remove HTML tags
    message = BeautifulSoup(message, "html.parser").get_text()

    # Remove URLs
    message = re.sub(r'http\S+|www\S+|https\S+', '', message, flags=re.MULTILINE)

    # Remove emojis
    message = emoji.demojize(message)
    message = re.sub(r':[a-z_&]+:', '', message)

    # Remove non-alphanumeric characters
    message = re.sub(r'\W+', ' ', message)

    # Convert to lowercase
    message = message.lower()

    # Remove stopwords and lemmatize
    # message = ' '.join(lemmatizer.lemmatize(word) for word in message.split() if word not in stop_words)

    # Spell check | Nah, typos are the essense of the data
    # misspelled = spell.unknown(message.split())
    # for word in misspelled:
    #     # Get the one 'most likely' answer
    #     correct = spell.correction(word)
    #     message = message.replace(word, correct)

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
    if BLOCK_EMBEDS and message['embeds'] != []:
        return False
    if USER is not None and message['author'] != USER:
        return False
    if len(list(message['content'])) < 10:
        return False
    # Ensure the message is in English
    # if detect(message['content']) != 'en':
    #     return False
    return True

import csv

def format_message(file_path):
    messages = []
    with open(file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)[:-VAL_LENGTH]
        for message in data:
            if not validate_message(message):
                continue
            author = message['author']
            if author != USER and USER is not None:
                continue
            content = clean_message(message['content'])
            messages.append([author, content])
    return messages

def format_messages(file_paths, output_path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(format_message, file_path): file_path for file_path in file_paths}
        with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file)
            for future in concurrent.futures.as_completed(future_to_file):
                messages = future.result()
                writer.writerows(messages)

format_messages(USED_FILES, OUTPUTNAME)

with open(OUTPUTNAME, 'r', encoding='utf-8') as input_file:
    data = list(csv.reader(input_file))
    TOT_MESSAGES = len(data)
    print(f"Total messages: {TOT_MESSAGES}")

def create_validation_data():
    for file_path in USED_FILES:
        messages = []
        with open(file_path, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)
            total_messages = 0
            for message in data:
                author = message['author']
                content = message['content']
                messages.append([author, content])
                total_messages += 1
                if total_messages >= VAL_LENGTH:
                    break
    
    with open(VAL_OUTPUTNAME, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(messages)

create_validation_data()

# def format_message(file_path):
#     messages = []
#     with open(file_path, 'r', encoding='utf-8') as input_file:
#         data = json.load(input_file)[:-VAL_LENGTH]
#         for message in data:
#             if not validate_message(message):
#                 continue
#             author = message['author']
#             if author != USER and USER is not None:
#                 continue
#             content = clean_message(message['content'])
#             messages.append(f'{author}: {content} <|endoftext|>\n')
#     return messages

# def format_messages(file_paths, output_path):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_file = {executor.submit(format_message, file_path): file_path for file_path in file_paths}
#         with open(output_path, 'w', encoding='utf-8') as output_file:
#             for future in concurrent.futures.as_completed(future_to_file):
#                 messages = future.result()
#                 output_file.writelines(messages)

# format_messages(USED_FILES, OUTPUTNAME)

# with open(OUTPUTNAME, 'r', encoding='utf-8') as input_file:
#     data = input_file.readlines()
#     TOT_MESSAGES = len(data)
#     print(f"Total messages: {TOT_MESSAGES}")


# def create_validation_data():
#     for file_path in USED_FILES:
#         messages = []
#         with open(file_path, 'r', encoding='utf-8') as input_file:
#             data = json.load(input_file)
#             total_messages = 0
#             for message in data:
#                 author = message['author']
#                 content = message['content']
#                 messages.append(f'{author}: {content}\n')
#                 total_messages += 1
#                 if total_messages >= VAL_LENGTH:
#                     break
    
#     with open(VAL_OUTPUTNAME, 'w', encoding='utf-8') as output_file:
#         output_file.writelines(messages)

# create_validation_data()