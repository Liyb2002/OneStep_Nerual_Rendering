import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Load model from final training checkpoint
model_path = "checkpoints/outputs/checkpoint-60"  # Directly use trained model
dataset_path = "dataset/formatted_data/training_dataset.json"  # Path to dataset

print(f"🚀 Loading fine-tuned model from local: {model_path} ...")

# 🔹 Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Ensure correct precision
    device_map="auto",
    low_cpu_mem_usage=True  # Reduce memory usage
)

print("✅ Model loaded successfully! Ready for inference.")

# ====== 🔹 Define the Prompt Format ====== #
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the input sketch data and analyze the spatial relationships step by step.

### Instruction:
You are an expert in geometric reasoning and sketch analysis. Your task is to analyze the provided stroke cloud and determine the appropriate construction lines that should be added.

### Stroke Cloud Data:
{}

### Response:
<think>{}"""

# ====== 🔹 Load Dataset & Select a Sample Input ====== #
print(f"📂 Loading dataset from: {dataset_path} ...")

with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Ensure dataset structure is correct
if "train" in dataset:
    dataset = dataset["train"]

sample_entry = random.choice(dataset)  # Randomly pick one sample
input_data = sample_entry["input"]  # Extract input JSON from dataset

print("\n🎯 **Sampled Input from Dataset:**")
print(json.dumps(input_data, indent=2))  # Pretty-print JSON input (FIXED)

# 🔹 Format Input for the Model
formatted_input = prompt_style.format(json.dumps(input_data), "")

# 🔹 Tokenize & Run Inference
print("\n🚀 Generating response...")
inputs = tokenizer([formatted_input], return_tensors="pt").to("cuda")  # Change to "cpu" if needed

with torch.no_grad():  # Optimize inference
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,  # Increase if response is too short
        use_cache=True,
    )

# 🔹 Decode & Print Response
response = tokenizer.batch_decode(outputs)
generated_text = response[0].split("### Response:")[1].strip()

print("\n🚀 **Generated Construction Lines:**")
print(generated_text)
