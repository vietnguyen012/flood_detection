import torch
from sklearn.model_selection import train_test_split

from data import TextDataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from utils import apk
def compute_metrics(eval_pred):
    pred = torch.Tensor(eval_pred[0].squeeze(-1))
    ids = torch.arange(eval_pred[1].shape[0])
    labels = torch.tensor(eval_pred[1].squeeze(-1))
    mask = labels == 1
    valid_id = ids[mask].tolist()
    confidences, sorted_idx  = pred.sort(0, descending=True)
    sorted_idx = sorted_idx.tolist()
    ids = ids.tolist()
    id_imgs = [ids[i] for i in sorted_idx]
    return {"eval_ap300":apk(valid_id, id_imgs, k=300)}

data = TextDataset("./data/dev", "images", None, "devset_images_gt.csv","devset_images_metadata.json", "bert-base-uncased")
train_data,val_data = train_test_split(data)
model_checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)

metric_name = "ap300"
model_name = model_checkpoint.split("/")[-1]
task = "classification"
batch_size=16
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=200,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=train_data.tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
