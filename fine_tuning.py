import json
import os

# Define paths
data_path = os.path.join(os.getcwd(), 'annotated_input')
formatted_data_path = os.path.join(os.getcwd(), 'dataset', 'formatted_data')

def find_second_level_folders(root_path):
    """
    Recursively find all second-level folders inside root_path.
    
    Parameters:
    - root_path: The main directory containing multiple nested folders.

    Returns:
    - List of second-level folder paths.
    """
    second_level_folders = []
    
    for first_level in os.listdir(root_path):
        first_level_path = os.path.join(root_path, first_level)
        
        if os.path.isdir(first_level_path):  # Ensure it's a folder
            for second_level in os.listdir(first_level_path):
                second_level_path = os.path.join(first_level_path, second_level)
                if os.path.isdir(second_level_path):  # Ensure it's a folder
                    second_level_folders.append(second_level_path)
    
    return second_level_folders

def prepare_training_data(data_path, formatted_data_path):
    """
    Reads input.json and gt_output.json from all second-level folders in data_path
    and formats them for DeepSeek fine-tuning.
    
    Saves:
    - training_dataset.json inside formatted_data_path.
    """
    training_data = []
    
    # Find all second-level subfolders
    second_level_folders = find_second_level_folders(data_path)

    for folder in second_level_folders:
        input_path = os.path.join(folder, "input.json")
        output_path = os.path.join(folder, "gt_output.json")

        if os.path.exists(input_path) and os.path.exists(output_path):
            with open(input_path, "r") as f:
                input_json = json.load(f)
            with open(output_path, "r") as f:
                output_json = json.load(f)

            training_data.append({
                "input": f"Given the stroke cloud data:\n{json.dumps(input_json, indent=2)}, what are the construction line relations?",
                "output": json.dumps(output_json, indent=2)
            })

    # Ensure the formatted_data_path exists
    if not os.path.exists(formatted_data_path):
        os.makedirs(formatted_data_path)

    # Save formatted dataset inside formatted_data_path
    dataset_path = os.path.join(formatted_data_path, "training_dataset.json")

    with open(dataset_path, "w") as f:
        json.dump(training_data, f, indent=4)

    print(f"Formatted training dataset saved at: {dataset_path}")

# Run the function using the updated paths
prepare_training_data(data_path, formatted_data_path)
