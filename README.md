### the models are training 2 gpus tesla v100 using distributed training strategy.

### To train text-image fused multi-modal model run:
```
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env train.py
```
### To train text model run: 
```
python -m torch.distributed.launch --nproc_per_node 2 text_train.py
```
### To train image model run:
```
cd pytorch-image-models
./distributed_train.sh 2 data -d custom_dataset --model gluon_xception65 --bce-loss --sched cosine --epochs 1000 --warmup-epochs 2 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 140 --amp -j 2
```

### Result on random train val split: 
the Fusion model uses bert-base-uncased and ViT acheived the best out of three with ap@300: 0.88
Image model with xceptions architecture achieved acc@1:0.84
Text model uses bert-base-uncased achieved ap@300:0.23

### run prediction with fusion model:
```
python test.py
```
