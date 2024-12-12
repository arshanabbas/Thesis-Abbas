import os
import json

# Input and output paths
json_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/new_result.json"
output_json_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Output/updated_result.json"

# Renaming pattern
image_rename_pattern = "image_{:03d}.jpg"

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Save JSON file
def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure output directory exists
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Update JSON annotations to match renamed files
def update_annotations(json_data):
    id_mapping = {}

    # Update the images section
    for idx, image_entry in enumerate(json_data["images"]):
        old_id = image_entry["id"]
        new_file_name = image_rename_pattern.format(idx + 1)
        image_entry["file_name"] = new_file_name

        # Map old ID to new sequential ID
        id_mapping[old_id] = idx
        image_entry["id"] = idx

    # Update the annotations section
    for annotation in json_data["annotations"]:
        if annotation["image_id"] in id_mapping:
            annotation["image_id"] = id_mapping[annotation["image_id"]]

    return json_data

# Main function
if __name__ == "__main__":
    # Load the original annotations
    annotations = load_json(json_path)

    # Update the annotations
    updated_annotations = update_annotations(annotations)

    # Save the updated annotations
    save_json(updated_annotations, output_json_path)

    print(f"Updated annotations saved to {output_json_path}")