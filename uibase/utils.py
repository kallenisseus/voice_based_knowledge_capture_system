import json
import os
from django.conf import settings

def read_json_data():
    # Put your JSON file in the project root or specify the path
    file_path = os.path.join(settings.BASE_DIR,'data', 'database.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []