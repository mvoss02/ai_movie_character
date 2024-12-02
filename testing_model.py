from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch
from trl import setup_chat_format
from helper import config

# Define the path to the fine-tuned model
peft_model_id = "./output/model"  # Your fine-tuned model path

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16  # Change from Niklas / Different from Phil Schmid's blog post
    )

# Load the model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16, quantization_config=bnb_config,)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# Set model to evaluation mode (important for inference)
model.eval()

# Tokenize the input
messages = [
    {
        "role": "system",
        "content": "You are Patrick Bateman, a narcissist working on Wall Street as a stockbroker at Pierce & Pierce.",
    },
    {"role": "user", "content": "How do you get such good skin?"},
]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
input_ids = tokenized_chat.to('cuda')

# Generate the response
output = model.generate(
    input_ids=input_ids,
    max_length=128,  # Limit output length to a single response
    temperature=0.8,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1
)

# Decode and print the result
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:", response)