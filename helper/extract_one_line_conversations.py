import os
import re
import random
import PyPDF2
import itertools
from helper import config


character_lines = []

def convert_pdf_to_txt(path_to_pdf: str, path_to_output: str):
    pdf = open(path_to_pdf, 'rb')
    pdf = PyPDF2.PdfReader(pdf)
    
    print('Path to CSV:', path_to_pdf)
    print('Number of pages of pdf:', len(pdf.pages))
    
    conversations_by_page = []
    
    for page in pdf.pages:
        text = page.extract_text()
        
        # Remove whitespace elements in list
        page_list = [line for line in text.splitlines() if line.strip()]
        
        conversations_by_page.append(page_list)
    
    # Remove empty lists
    conversations_by_page = [el for el in conversations_by_page if el]
    
    # Combine all conversations to one list
    all_conversations = list(itertools.chain.from_iterable(conversations_by_page))
    
    with open(f"{path_to_output}", "w", encoding="utf-8") as file:
        for item in all_conversations:
            file.write(item + "\n")

def strip_parentheses(s):
    return re.sub(r'\(.*?\)', '', s)
    
def is_single_word_all_caps(s):
    # First, we split the string into words
    words = s.split()

    # Check if the string contains only a single word
    if len(words) != 1:
        return False

    # Make sure it isn't a line number
    if bool(re.search(r'\d', words[0])):
        return False

    # Check if the single word is in all caps
    return words[0].isupper()
    
def process_directory(directory_path, character_name):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ignore directories
            extract_character_lines(file_path, character_name)
            
    with open(f'./{character_name}_lines.jsonl', 'w', newline='') as outfile:
        prevLine = ''
        for s in character_lines:
            if (s.startswith('DATA:')):
                outfile.write("{\"messages\": [{\"role\": \"system\", \"content\": \"Data is an android in the TV series Star Trek: The Next Generation.\"}, {\"role\": \"user\", \"content\": \"" + prevLine + "\"}, {\"role\": \"assistant\", \"content\": \"" + s + "\"}]}\n")
            prevLine = s

def extract_character_lines(file_path, character_name):
    with open(file_path, 'r') as script_file:
        lines = script_file.readlines()

    is_character_line = False
    current_line = ''
    current_character = ''
    for line in lines:
        strippedLine = line.strip()
        if (is_single_word_all_caps(strippedLine)):
            is_character_line = True
            current_character = strippedLine
        elif (line.strip() == '') and is_character_line:
            is_character_line = False
            dialog_line = strip_parentheses(current_line).strip()
            dialog_line = dialog_line.replace('"', "'")
            character_lines.append(current_character + ": " + dialog_line)
            current_line = ''
        elif is_character_line:
            current_line += line.strip() + ' '
            
def split_file(input_filename, train_filename, eval_filename, split_ratio=0.8, max_lines=10000):
    """
    Splits the lines of the input file into training and evaluation files.

    :param input_filename: Name of the input file to be split.
    :param train_filename: Name of the output training file.
    :param eval_filename: Name of the output evaluation file.
    :param split_ratio: Ratio of lines to be allocated to training. Default is 0.8, i.e., 80%.
    """
    
    with open(input_filename, 'r') as infile:
        lines = infile.readlines()

    # Shuffle lines to ensure randomness
    random.shuffle(lines)
    
    lines = lines[:max_lines]

    # Calculate the number of lines for training
    train_len = int(split_ratio * len(lines))

    # Split the lines
    train_lines = lines[:train_len]
    eval_lines = lines[train_len:]

    # Write to the respective files
    with open(train_filename, 'w') as trainfile:
        trainfile.writelines(train_lines)

    with open(eval_filename, 'w') as evalfile:
        evalfile.writelines(eval_lines)

process_directory('/home/mvoss/projects/conversational_chatbot/test/movies', 'BATEMAN')
#split_file('./bateman_lines.jsonl', './bateman_train.jsonl', './bateman_eval.jsonl')

#for input_file, output_file in config.MOVIE_SCRIPT_PDF_TO_TEXT.items():
#    convert_pdf_to_txt(path_to_pdf=input_file, path_to_output=output_file)

