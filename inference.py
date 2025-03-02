import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Load model from final training checkpoint
model_path = "checkpoints/outputs/checkpoint-690"  # Directly use trained model
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
Write a response that follows the **exact** required format: ((relation, stroke_id), (relation, stroke_id)). 
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
((relation, stroke_id), (relation, stroke_id))

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
# question = {
#     "strokes": [
#         {"id": 1, "type": "feature_line", "geometry": [[0, 0, 0], [1, 0, 0]]},
#         {"id": 2, "type": "feature_line", "geometry": [[1, 0, 0], [1, 1, 0]]}
#     ],
#     "intersections": [[1, 2]]
# }

# # Convert question to JSON string
# question_json = json.dumps(question, indent=4)

# # Format the prompt properly
# formatted_prompt = prompt_style.format(question_json, "")

# # Tokenize input
# inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

# # Generate output
# outputs = model.generate(
#     input_ids=inputs.input_ids,
#     attention_mask=inputs.attention_mask,
#     max_new_tokens=1200,
#     use_cache=True,
# )

# # Decode response
# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# # Extract model response after "### Response:"
# if "### Response:" in response:
#     response = response.split("### Response:")[1].strip()

# print("Sample Model Output:", response)



# ====== 🔹 Load Dataset & Select a Sample Input ====== #
print(f"📂 Loading dataset from: {dataset_path} ...")

with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Ensure dataset structure is correct
if "train" in dataset:
    dataset = dataset["train"]

sample_entry = random.choice(dataset)  # Randomly pick one sample
input_data = sample_entry["input"]  # Extract input JSON from dataset

# 🔹 Format Input for the Model (COMPACT JSON)
formatted_input = prompt_style.format(json.dumps(input_data, separators=(',', ':')), "")

# 🔹 Ensure Input is Clean (No Newlines or Extra Spaces)
formatted_input = formatted_input.replace("\n", " ").strip()

# 🔹 Tokenize & Run Inference
print("\n🚀 **Formatted Input Sent to Model:**")

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
generated_text = response[0].split("### Response:")[1].strip() if "### Response:" in response[0] else response[0]

print("\n🚀 **Generated Construction Lines:**")
print(generated_text)
