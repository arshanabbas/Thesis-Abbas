import os
import json

# Input and output paths
input_json_path ="F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/new_result.json"
output_json_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Output/updated_result.json"

# Load JSON data
with open(input_json_path, 'r') as file:
    data = json.load(file)

# Update 'file_name' sequentially
for idx, img in enumerate(data.get('images', [])):
    new_file_name = f"image_{idx + 1:03}.jpg"
    img['file_name'] = new_file_name

# Save the updated JSON file
with open(output_json_path, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Updated file names saved to '{output_json_path}'")