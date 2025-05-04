import json, os

# 1. Load your gender map
with open("gender_map.json", "r") as f:
    gender_map = json.load(f)

# 2. Prepare empty splits
splits = {"train": [], "test": []}

# 3. For each subject folder, sort images and assign 1–7 → train, 8–10 → test
for subj, gender in gender_map.items():
    folder = os.path.join("faces_dataset", subj)
    imgs = sorted(os.listdir(folder), key=lambda fn: int(fn.split('.')[0]))
    for idx, fn in enumerate(imgs, start=1):
        entry = {"path": os.path.join(folder, fn), "label": gender}
        if idx <= 7:
            splits["train"].append(entry)
        else:
            splits["test"].append(entry)

# 4. Write out splits.json
with open("splits.json", "w") as f:
    json.dump(splits, f, indent=2)

print(f"Train: {len(splits['train'])},  Test: {len(splits['test'])}")
