import json

json_file_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/new_result.json"

# Load JSON
with open(json_file_path, 'r') as file:
    data = json.load(file)

# List file names in the JSON
print("File names in the JSON:")
for img in data.get('images', []):
    print(img['file_name'])