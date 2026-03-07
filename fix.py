import json

def remove_embeddings(data):
    """Recursively remove 'embedding' keys from nested dictionaries"""
    if isinstance(data, dict):
        # Create a new dict without 'embedding' key
        return {k: remove_embeddings(v) for k, v in data.items() if k != 'embedding'}
    elif isinstance(data, list):
        # Process each item in list
        return [remove_embeddings(item) for item in data]
    else:
        # Return the value as-is
        return data

# Read the original JSON file
with open('database.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Remove all embeddings
cleaned_data = remove_embeddings(original_data)

# Save to a new JSON file
with open('data_no_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print("✓ Embeddings removed successfully!")
print(f"✓ Saved to: data_no_embeddings.json")