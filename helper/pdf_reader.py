import PyPDF2
import regex as re
import itertools
import json
from helper import config

def read_pdf(path_to_pdf: str, save_output: bool, path_to_output: str, main_character: str):
    pdf = open(path_to_pdf, 'rb')
    pdf = PyPDF2.PdfReader(pdf)
    
    print('Path to CSV:', path_to_pdf)
    print('Number of pages of pdf:', len(pdf.pages))
    
    conversations_by_page = []
    
    for page in pdf.pages:
        text = page.extract_text()
        
        #print('Step 1: Remove text within parentheses')
        text_no_parentheses = re.sub(r"\([^)]*\)", "", text)
        
        #print('Step 2: Replace character names surrounded by newlines with <NAME>')
        text_with_names = re.sub(r'(?<=\n)([A-Z]+)(?=\n)', r'<\1>', text_no_parentheses)
        
        #print('Step 3: Remove mid sentence new lines')
        text_no_mid_sentence_newlines = re.sub(r'(?<![.?!])\n(?!\s)', ' ', text_with_names)
        
        #print('Step 4: Remove newlines following a <NAME> pattern')
        text_no_newlines_after_name = re.sub(r'(<[A-Z]+>)\n', r'\1', text_no_mid_sentence_newlines)
        
        #print('Step 5: Keep only the <NAME> followed by the dialogue up until the next newline')
        text_only_name_dialouge = re.sub(r'(<[A-Z]+>.*?)\n', r'\1\n', text_no_newlines_after_name)
        
        #print('Step 6: Remove everything except lines with <NAME> followed by text until newline')
        # This regex keeps only the lines where there's a name followed by dialogue up to the newline
        text_spaces = re.sub(r'^(?!<.*?>).*$', '', text_only_name_dialouge, flags=re.M)
        
        #print('Step 7: Replace multiple spaces with a single space')
        text_final = re.sub(r'\s+', ' ', text_spaces)
        
        #print('Step 8: Add a newline before each <NAME>')
        text_final_with_newlines = re.sub(r'(<[A-Z]+>)', r'\n\1', text_final)
        
        #print('Step 9: Remove the last three characters if they contain a number (1 to 1000) followed by a dot')
        text_final_no_last_number = re.sub(r'(\d{1,3})\.$', '', text_final_with_newlines)
        
        # print('Step 10: Replace the main character's name with <NAME> to generlaise between various movie scripts')
        text_final = re.sub(rf"<{main_character}>", "<MAIN>", text_final_no_last_number)
        
        # Remove whitespace elements in list
        page_list = [line for line in text_final.splitlines() if line.strip()]
        
        conversations_by_page.append(page_list)
    
    # Remove empty lists
    conversations_by_page = [el for el in conversations_by_page if el]
    
    # Combine all conversations to one list
    all_conversations = list(itertools.chain.from_iterable(conversations_by_page))
    
    if save_output:
        # Save to a text file
        with open(f"{path_to_output}/{path_to_pdf[14:-4]}.txt", "w", encoding="utf-8") as file:
            for item in all_conversations:
                file.write(item + "\n")
         
    return all_conversations

def convert_roles(string: str) -> dict[str, str]:
    # Regular expression to extract the text between <>
    match = re.search(r"<(.*?)>", string)
    
    # Handle case where no match is found
    if not match:
        return {}
    
    # Extract the matched text (role)
    role = match.group(1)
    
    # Replace the match with nothing
    modified_text = re.sub(r"<.*?>", "", string).strip()
    
    return {'role': role, 'content': modified_text}

def convert_txt_to_json(path_to_txt: str, path_to_output: str) -> None:
    with open(path_to_txt, 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    extarcted_role = []
    for line in content:
        extarcted_role.append(convert_roles(line))
        
    
    # Save data to a JSONL file
    with open(f"{path_to_output}/{path_to_txt[21:-4]}.txt", 'w', encoding='utf-8') as f:
        for entry in extarcted_role:
            # Serialize each dictionary to JSON and write it as a single line
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')  # Add a newline to separate each JSON object
            
def create_conversation(sample):
    """
    Convert a dictionary of messages into a standardized conversation format where:
    - 'MAIN' is mapped to 'assistant'.
    - All other roles are mapped to 'user'.
    
    Args:
    sample (dict): A dictionary containing a conversation history in the "messages" field.

    Returns:
    dict: A dictionary in the format expected by LLMs, with normalized roles.
    """
    # Define the system message
    system_message = {
        "role": "system",
        "content":  config.SYSTEM_MESSAGE
    }
    
    # Start with the system message
    processed_messages = [system_message]
    
    for message in sample["messages"]:
        if message["role"] == "MAIN":
            role = "assistant"
        else:
            role = "user"
        
        processed_messages.append({
            "role": role,
            "content": message["content"]
        })
    
    return {"messages": processed_messages}

def convert_pdf_to_txt(path_to_pdf: str, path_to_output: str):
    # Open the PDF file
    with open(path_to_pdf, 'rb') as pdf_file:
        pdf = PyPDF2.PdfReader(pdf_file)
        
        print('Path to PDF:', path_to_pdf)
        print('Number of pages in PDF:', len(pdf.pages))
        
        # Initialize a single string to hold all text
        all_text = ""
        
        # Iterate through all pages and extract text
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # Check if the page has any text
                all_text += text.strip() + "\n"  # Add a newline between pages
        
        # Write the combined text to the output file
        with open(path_to_output, "w", encoding="utf-8") as file:
            file.write(all_text)
            
def parse_dialogues(file_path: str):
    dialogues = []  # List to store parsed dialogue tuples (name, text)
    current_speaker = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Remove surrounding whitespace
            
            if line.isupper():  # Check if the line is all uppercase
                # Save the previous dialogue if it exists
                if current_speaker and current_text:
                    dialogues.append((current_speaker, " ".join(current_text)))
                # Start a new dialogue
                current_speaker = line
                current_text = []
            elif line:  # If the line is not empty, add it to the current text
                current_text.append(line)
        
        # Add the last dialogue block after exiting the loop
        if current_speaker and current_text:
            dialogues.append((current_speaker, " ".join(current_text)))
    
    return dialogues


def conevrt_to_jsonl(file_path: str, main_character: str):
    dialogues = parse_dialogues(file_path)
    
    # Open the output file
    with open(f'./movie_scripts/final_scripts/{main_character}_lines.jsonl', 'w', newline='', encoding='utf-8') as outfile:
        i = 0
        while i < len(dialogues):
            messages = []

            # Add system message
            messages.append({
                'role': 'system',
                'content': 'You are Patrick Bateman, a narcissist working on Wall Street as a stockbroker at Pierce & Pierce.'
            })

            # Check if current dialogue belongs to main character
            if dialogues[i][0].startswith(f'{main_character}'):
                # Include previous line if not from the main character
                if i > 0 and not dialogues[i-1][0].startswith(f'{main_character}'):
                    messages.append({
                        "role": "user",
                        "content": dialogues[i - 1][1]
                    })
                
                # Collect dialogue block
                while i < len(dialogues) and dialogues[i][0].startswith(f'{main_character}'):
                    messages.append({
                        "role": "assistant",
                        "content": dialogues[i][1]
                    })
                    i += 1

                # Write to JSONL
                jsonl_entry = {"messages": messages}
                outfile.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
            else:
                i += 1