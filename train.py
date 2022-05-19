from apex import amp
from data import TextImageDataset
from torchvision import transforms
from model import FusionLayer
from sam import SAM
import torch, gc
import torch.nn as nn
from stepLR import StepLR
from torch.optim import lr_scheduler
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from model import L2_norm, TripletLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import random
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 8  # 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # True
from torch.utils.data import DataLoader, RandomSampler,  SequentialSampler
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--world_size', type=int, help="num of processes")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
from utils import *
from sklearn.model_selection import train_test_split

def eval(model,val_loader,parallel,distributed):
    flood_id = []
    id_imgs = []
    confidences = []
    for step, batch in enumerate(val_loader):
        batch = {k:batch[k].to(device,non_blocking=True) if k !="id" else batch[k] for k, v in batch.items()}
        id_img = batch["id"]
        labels = batch["label"].squeeze()
        idx_flood = torch.nonzero(labels).squeeze()
        flood_id.extend([id_img[i] for i in idx_flood.detach().tolist()])
        with torch.no_grad():
            if parallel or distributed:
                _,_, confidence =  model(img_tensor= batch["img_tensor"], input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],label=batch["label"])
            else:
                _, confidence = model(img_tensor= batch["img_tensor"], input_ids=batch["input_ids"],
                               attention_mask= batch["attention_mask"],label=batch["label"])
        id_imgs.extend(id_img)
        confidences.extend(confidence)
    confidences = torch.tensor(confidences)
    confidences, sorted_idx = confidences.sort(0, descending=True)
    sorted_idx = sorted_idx.tolist()
    id_imgs = [id_imgs[i] for i in sorted_idx]
    return apk(flood_id, id_imgs, k=300)



train_transf = val_transf = transforms.Compose([
    transforms.Resize(size=248, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5000, 0.5000, 0.5000], [0.5000, 0.5000, 0.5000])]
)

distributed = True
parralel = False
batch_size = 50
best_ap = -1
rho = 0.05
learning_rate = 1e-1
epochs=300
use_normal_opt= True
use_sam = False
max_norm = 2
device =None
nbs = 64  # nominal batch size
accumulate = max(round(nbs / batch_size), 1)
weight_decay=5e-4
weight_decay *= batch_size * accumulate / nbs
warmup_bias_lr= 0.1
warmup_momentum = 0.8
momentum = 0.937
workers = 3
last_opt_step = -1
nd = torch.cuda.device_count()
nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])


if distributed:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK'))
    device = torch.device(f'cuda:{local_rank}')
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', -1))
    torch.cuda.set_device(device)
    global_rank = dist.get_rank()
    with torch_distributed_zero_first(rank):
        train_data = TextImageDataset("./data/dev", "images", train_transf, "devset_images_gt.csv",
                                      "devset_images_metadata.json", "bert-base-uncased")
        pos_weight = train_data.pos_weight

        train_data,val_data = train_test_split(train_data,test_size=0.2)

    model = FusionLayer("bert-base-uncased", "ViT", use_triplet=True, pos_weight=pos_weight,distributed=True).to(device)
    if use_sam == True:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay)
        schedular = StepLR(optimizer, learning_rate, epochs)
    else:
        optimizer = get_optimizer(model)
        lf = one_cycle(1, 0.1, epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
        os.environ.get('WORLD_SIZE'),
        os.environ.get('RANK'),
        os.environ.get('LOCAL_RANK'),
        os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
else:
    local_rank = None
    global_rank = -1
    train_data = TextImageDataset("./data/dev", "images", train_transf, "devset_images_gt.csv",
                                  "devset_images_metadata.json", "roberta-base")
    pos_weight = train_data.pos_weight

    train_data, val_data = train_test_split(train_data, test_size=0.2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FusionLayer("bert-base-uncased", "ViT", use_triplet=True, pos_weight=pos_weight).to(device)
    if use_sam == True:
        momentum = 0.9
        weight_decay = 5e-4
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay)
        schedular = StepLR(optimizer, learning_rate, epochs)
    else:
        optimizer = get_optimizer(model)
        lf = one_cycle(1, 0.1, epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

if distributed:
    sampler = DistributedSampler(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size= batch_size // world_size,sampler=sampler,num_workers=nw,pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size // world_size * 2, drop_last=False, pin_memory=True)
else:
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, drop_last=False)

total = len(train_loader)

cuda = device.type != 'cpu'
if parralel:
    model = nn.DataParallel(model)

if parralel or distributed:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    triploss = TripletLoss(device)
    l2 = L2_norm()
if global_rank == 0 or global_rank == -1:
    print("====================Begin Training==========================")
for epoch in range(epochs):
    mean_loss = []
    model.train(True)
    for step, batch in enumerate(train_loader):
        ni = step + total * epoch
        batch = {k:batch[k].to(device,non_blocking=True) if k !="id" else batch[k] for k, v in batch.items()}
        if use_normal_opt:
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, warmup_momentum, momentum)

        if use_normal_opt == True:
            if parralel or distributed:
                output, label = model(img_tensor=batch["img_tensor"], input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"], label=batch["label"])

            else:
                loss, _ = model(img_tensor=batch["img_tensor"], input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"], label=batch["label"])
            if global_rank != -1:
                norm_out = l2(output)
                loss = criterion(output.float(), label.float()) + triploss(norm_out.float(),label.squeeze(-1).float())
            if global_rank != -1:
                loss*=world_size
            loss.backward()

            if ni - last_opt_step >= accumulate:
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if not parralel:
                    model.zero_grad()
                optimizer.zero_grad()
                if distributed:
                    if step%10==0 and global_rank==0:
                        print('Epoch: {} step: {} loss: {}'.format(epoch,step, loss.item()))
            last_opt_step = ni
            mean_loss.append(loss.item())
        else:
            if parralel or distributed:
                #pass1
                output, label = model(img_tensor=batch["img_tensor"], input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"], label=batch["label"])
                norm_out = l2(output)
                loss = criterion(output.float(), label.float()) + triploss(norm_out.float(),
                                                                           label.squeeze(-1).float())
                if rank != -1:
                    loss *= world_size

                loss.backward()
                optimizer.first_step()
                #pass2
                output, label = model(img_tensor=batch["img_tensor"], input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"], label=batch["label"])
                norm_out = l2(output)
                loss_prime = criterion(output.float(), label.float()) + triploss(norm_out.float(),
                                                                           label.squeeze(-1).float())
                if rank != -1:
                    loss_prime *= world_size

                loss_prime.backward()
                optimizer.second_step()
                schedular(epoch)
                if not parralel:
                    model.zero_grad()
                mean_loss.append(loss.item())
    if use_normal_opt:
        scheduler.step()
    model.train(False)
    if distributed and global_rank == 0 or global_rank==-1:
        ap = eval(model, val_loader, parralel, distributed)
        print(f"Epoch {epoch} -ap {ap} - step {step}/{total} - loss: {sum(mean_loss)/len(mean_loss)}")
        if ap > best_ap or (epoch > 1 and epoch < 6):
            best_ap = ap
            torch.save(model.module.state_dict(), f"./models/model_epoch_{epoch}_loss_{sum(mean_loss)/len(mean_loss)}_ap_{ap}.bin")

