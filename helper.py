import json

def read_json(file_path):
    """
    Reads a JSON file and returns its contents.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None
