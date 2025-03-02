import os
from huggingface_hub import login
from datasets import load_dataset
import wandb
from unsloth import FastLanguageModel

import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()  # Forces GPU garbage collection

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ====== 🔹 Hugging Face Authentication ====== #
token_path = "creds/hf_token.txt"
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        hf_token = f.read().strip()  # Read and remove any extra spaces/newlines
    login(hf_token)
    print("✅ Successfully logged in to Hugging Face!")
else:
    raise FileNotFoundError(f"❌ Hugging Face token not found at {token_path}")

# ====== 🔹 Weights & Biases Authentication ====== #
wandb_path = "creds/wandb.txt"
if os.path.exists(wandb_path):
    with open(wandb_path, "r") as f:
        wb_token = f.read().strip()
    wandb.login(key=wb_token)
    run = wandb.init(
        project="Fine-tune-DeepSeek-R1 on Construction Lines",
        job_type="training",
        anonymous="allow",
    )
    print("✅ Successfully logged into Weights & Biases!")
else:
    print("⚠️ Warning: W&B token not found. Skipping W&B logging.")

# ====== 🔹 Load Model & Tokenizer ====== #
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-14B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token,
)

# ====== 🔹 Define Prompt Style for Fine-Tuning ====== #
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that follows the **exact** required format: [[relation, stroke_id], [relation, stroke_id]]. 
Do not give any explanation

### **Instruction**:
You are an expert in **geometric reasoning and sketch analysis**. Your task is to analyze the given **stroke cloud** and determine the appropriate **construction lines** that should be added based on stroke relationships.

### **Guidelines for Construction Line Prediction**:
- **Midpoint Alignment**: Create construction lines connecting the midpoints of intersecting strokes.
- **Extension Alignment**: If a stroke naturally extends toward another, define a construction line guiding this alignment.
- **Projection Alignment**: If a stroke belongs to a curve, project its key points to maintain geometric consistency.
- **Scaffolding Support**: Ensure structural alignment by connecting stroke endpoints to key intersections.

### **Input Format**:
The input consists of:
- **strokes**: A list of strokes where:
  - `"id"`: A unique identifier for each stroke.
  - `"type"`: `"line"` for straight strokes, `"curve"` for curved strokes.
  - `"coords"`: A list of **six numerical values** representing the **start and end points** in 3D space.
- **intersections**: A list indicating pairs of stroke IDs that intersect.

### **Expected Output Format**:
[[relation, stroke_id], [relation, stroke_id]]

### **Explanation of Output Format**:
Each **construction line** is defined by **two points**, each associated with a stroke.
Each one can be written as (relation, stroke_id), which is the current point's relation to a stroke in the given input.
 **relation** can be:
- `"midpoint"`: The point is the **midpoint** of the stroke.
- `"on_extension"`: The point lies **on the extension** of the stroke.
- `"endpoint"`: The point is an **endpoint** of the stroke.

---

### **Stroke Cloud Data**:
{}

### **Response**:
{}
"""

# ====== 🔹 Test Prompt Before Training ====== #
# question = """{
#     "strokes": [
#         {"id": 1, "type": "feature_line", "geometry": [[0,0,0], [1,0,0]]},
#         {"id": 2, "type": "feature_line", "geometry": [[1,0,0], [1,1,0]]}
#     ],
#     "intersections": [[1, 2]]
# }"""

# FastLanguageModel.for_inference(model)
# inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
# outputs = model.generate(
#     input_ids=inputs.input_ids,
#     attention_mask=inputs.attention_mask,
#     max_new_tokens=1200,
#     use_cache=True,
# )
# response = tokenizer.batch_decode(outputs)
# print("Sample Model Output:", response[0].split("### Response:")[1])

# # ====== 🔹 Load Dataset for Fine-Tuning ====== #
dataset_path = "dataset/formatted_data/training_dataset.json"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

dataset = load_dataset("json", data_files={"train": dataset_path})
dataset = dataset["train"]

EOS_TOKEN = "</s>"  # Define the end-of-sequence token

# ====== 🔹 Format Dataset for Training ====== #
def formatting_prompts_func(examples):
    inputs = examples["input"]  # Ensure key names match your dataset
    outputs = examples["output"]
    texts = []

    for input_text, output_text in zip(inputs, outputs):
        text = prompt_style.format(input_text, output_text) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)


print("-------------SET UP LoRA Fine Tuning---------------------")


num_samples = len(dataset)  # Total dataset size
effective_batch_size = 2 * 8  # batch_size * gradient_accumulation_steps
steps_per_epoch = num_samples // effective_batch_size
max_steps = steps_per_epoch * 5  # Train for 5 epochs

print(f"Dataset size: {num_samples}, Training for {max_steps} steps (5 epochs)")


model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="checkpoints/outputs",
    ),
)


trainer_stats = trainer.train()


print("-------------Save Model---------------------")

import os

# 🔹 Define the save path
checkpoints_dir = "checkpoints"
new_model_local = os.path.join(checkpoints_dir, "Model")

# 🔹 Ensure checkpoints directory exists
os.makedirs(checkpoints_dir, exist_ok=True)

# 🔹 Save Model & Tokenizer
print(f"🚀 Saving fine-tuned model to {new_model_local} ...")
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)

# 🔹 Save merged model (16-bit for efficient inference)
trainer.model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit")

print(f"✅ Model successfully saved in {new_model_local}!")


# print("-------------Upload Model---------------------")
# new_model_online = "liyb2000/DeepSeek-R1-Legal-COT"
# model.push_to_hub(new_model_online)
# tokenizer.push_to_hub(new_model_online)
# model.push_to_hub_merged(new_model_online, tokenizer, save_method="merged_16bit")
