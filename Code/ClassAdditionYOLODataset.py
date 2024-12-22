import os

def update_ignored_class(annotation_dir, ignored_class_id, new_class_id):
    for annotation_file in os.listdir(annotation_dir):
        if not annotation_file.endswith(".txt"):
            continue
        
        file_path = os.path.join(annotation_dir, annotation_file)

        # Read file contents
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Update class IDs
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if int(parts[0]) == ignored_class_id:
                parts[0] = str(new_class_id)  # Update the class ID
            updated_lines.append(' '.join(parts))
        
        # Write updated contents back to the file
        with open(file_path, 'w') as file:
            file.write('\n'.join(updated_lines) + '\n')
    
    print(f"Updated ignored class ID {ignored_class_id} to valid class ID {new_class_id} in {annotation_dir}")

# Example usage
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/YOLOv8"
ignored_class_id = -1
new_class_id = 3

update_ignored_class(annotation_dir, ignored_class_id, new_class_id)
