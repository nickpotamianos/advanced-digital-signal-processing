import os
import json

# Path to the faces_dataset directory
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'faces_dataset')

split = {"train": [], "test": []}

for person in sorted(os.listdir(DATASET_DIR)):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    # Images 1-8 for train, 9-10 for test
    for i in range(1, 9):
        img_path = os.path.join('faces_dataset', person, f"{i}.pgm")
        split["train"].append(img_path)
    for i in range(9, 11):
        img_path = os.path.join('faces_dataset', person, f"{i}.pgm")
        split["test"].append(img_path)

with open('split.json', 'w') as f:
    json.dump(split, f, indent=2)

print("split.json created with train and test splits.")
