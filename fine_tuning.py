import os
from huggingface_hub import login

# Read the Hugging Face token from a file
token_path = "creds/hf_token.txt"
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        hf_token = f.read().strip()  # Read and remove any extra spaces/newlines
    login(hf_token)
    print("Successfully logged in to Hugging Face!")


