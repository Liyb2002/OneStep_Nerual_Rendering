import os
from huggingface_hub import login

# Read the Hugging Face token from a file
token_path = "creds/hf_token.txt"
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        hf_token = f.read().strip()  # Read and remove any extra spaces/newlines
    login(hf_token)
    print("Successfully logged in to Hugging Face!")


import wandb

wandb_path = "creds/wandb.txt"
if os.path.exists(wandb_path):
    with open(wandb_path, "r") as f:
        wb_token = f.read().strip()  # Read and remove any extra spaces/newlines

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Qwen-14B on Legal COT Dataset', 
    job_type="training", 
    anonymous="allow"
)


from unsloth import FastLanguageModel

max_seq_length = 2048 
dtype = None 
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hf_token, 
)

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the input sketch data and analyze the spatial relationships step by step.

### Instruction:
You are an expert in geometric reasoning and sketch analysis. Your task is to analyze the provided stroke cloud and determine the appropriate construction lines that should be added.

### Stroke Cloud Data:
{}

### Response:
<think>{}"""

question = """{
    "strokes": [
        {"id": 1, "type": "feature_line", "geometry": [[0,0,0], [1,0,0]]},
        {"id": 2, "type": "feature_line", "geometry": [[1,0,0], [1,1,0]]}
    ],
    "intersections": [[1, 2]]
}"""

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])
