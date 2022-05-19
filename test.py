from data import TextImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model import FusionLayer
import torch
import pandas as pd
import cv2
import numpy as np

mean = [0.5000, 0.5000, 0.5000]
std = [0.5000, 0.5000, 0.5000]
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

incep_trnasf =  transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

vi_transf = transforms.Compose([
    transforms.Resize(size=248, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5000, 0.5000, 0.5000], [0.5000, 0.5000, 0.5000])]
)
batch_size=32

test_data1 = TextImageDataset("./data/test","images",vi_transf,None,"testset_images_metadata.json","bert-base-uncased",is_test=True)
test_data2 = TextImageDataset("./data/test","images",incep_trnasf,None,"testset_images_metadata.json","roberta-base",is_test=True)

test_loader_1 = DataLoader(dataset=test_data1, batch_size=32,shuffle=False)
test_loader_2 = DataLoader(dataset=test_data2, batch_size=32,shuffle=False)

model1 = FusionLayer("bert-base-uncased", "ViT",use_triplet=True).to("cuda")
model1.load_state_dict(torch.load("./models/model_epoch_3_loss_0.17915517556332244_ap_0.8880913955994127.bin"))

models = [model1]
test_loaders = [test_loader_1]
for model in models:
    model.eval()

id_imgs = []
confidences = []
images_tensors = []
total_confidences = []
for ith,model in enumerate(models):
    confidences = []
    test_loader = test_loaders[ith]
    for step, batch in enumerate(test_loader):
        batch = {k: batch[k].to("cuda") if k != "id" else batch[k] for k, v in batch.items()}
        id_img = batch["id"]
        images_tensor = batch["img_tensor"]
        with torch.no_grad():
            img_tensor = batch["img_tensor"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            batch_logits = model(img_tensor=img_tensor,input_ids=input_ids,attention_mask=attention_mask)
            confidences.extend(batch_logits.squeeze(-1).detach().cpu().numpy())
            if ith == 0:
                id_imgs.extend(id_img)
                images_tensors.extend(images_tensor)
    del model,test_loader
    total_confidences.append(np.array(confidences))

avg_confidences = np.zeros_like(total_confidences[0])
for confidence in total_confidences:
    avg_confidences += confidence
avg_confidences/=len(models)
avg_confidences = torch.sigmoid(torch.tensor(avg_confidences)).numpy()
images_tensors = [images_tensor.unsqueeze(0) for images_tensor in images_tensors]
images_tensors = torch.cat(images_tensors).detach().cpu()
sorted_idx = np.argsort(avg_confidences)[::-1]
sorted_idx = sorted_idx.tolist()

id_imgs = [id_imgs[i] for i in sorted_idx]
images_tensors = images_tensors[torch.tensor(sorted_idx)]
for ind,images_tensor in enumerate(images_tensors):
    images_tensor = inv_normalize(images_tensor)
    cv2.imwrite(f"./checking/{ind}.png",images_tensor.permute(1,2,0).numpy()*255)

dict = {'Id': id_imgs}
df = pd.DataFrame(dict)

df.to_csv('sm-12.csv',index=False)
