import json
import pandas as pd
with open("./data/labeling/testset_images_metadata.json") as f:
    data = json.load(f)
    data_images = data["images"]
label_df = pd.read_csv("./data/labeling/label_sheet.csv")
orig_id, ref_id = label_df["orig_id"].tolist(),label_df["ref_id"].tolist()
orig_id = [id.split("_")[0] for id in orig_id]
new_data = []
for data_image in data_images:
    new_item = {}
    if data_image["image_id"] not in orig_id:
        continue
    for k,v in data_image.items():
        if k == "image_id":
            new_id = ref_id[orig_id.index(v)]
            new_item[k] = new_id
        else:
            new_item[k] = v
    new_data.append(new_item)

with open("./data/dev/devset_images_metadata.json") as f:
    train_data = json.load(f)

for new_item in new_data:
    train_data["images"].append(new_item)
print(len(train_data["images"]))
out_file = open("./data/dev/new_devset_images_metadata.json", "w")
json.dump(train_data, out_file, indent=6)
out_file.close()