# 9991project
## STEPS
1. train StyleGAN and export model weights 
2. train pSp
3. train AGE


### Step 1 :StyleGAN2-ADA
#### Training StyleGAN2-ADA
```
cd stylegan2-ada-pytorch; python train.py \
--gpus=1 \
--cond=1 \
--cfg=11gb-gpu \
--metrics=None \
--outdir=/path/to/results \
--data=/path/to/dataset/ \
--snap=$snapshot_count \
--resume=/pretrained/wikiart.pkl \
--augpipe=bg \
--initstrength=0 \
--gamma=50 \
--mirror=True \
--mirrory=False \
--nkimg=$train_count
```
#### export model weights 
```
cd stylegan2-ada-pytorch; python export_weights1.py /path/to/results/network-snapshot-XXX.pkl /path/to/styleGAN/network-snapshot-XXX.pt
```

### Step 2: pSp
#### Training the pSp Encoder
modified the dataset path in pixel2style2pixel-modified/configs/path_config.py
```
cd pixel2style2pixel-modified; python scripts/train.py \
--dataset_type=ffhq_encode_cond \
--exp_dir=/path/to/pretrained/pSp/checkpoints \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=ResNetGradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0 \
--moco_lambda=0.5 \
--c_dim=26 \ 
--output_size=512 \
--stylegan_weights=/path/to/styleGAN/network-snapshot-XXX.pt
```

### Step 3 :AGE 
#### get class embedding
```
python tools/get_class_embedding.py \
--class_embedding_path=/path/to/save/classs/embeddings \
--psp_checkpoint_path=/path/to/pretrained/pSp/checkpoints \
--train_data_path=/path/to/training/data \
--test_batch_size=4 \
--test_workers=4
```

#### Training AGE
modified the dataset path in AGE-modified/configs/path_config.py
```
cd AGE-modified;python -m torch.distributed.launch \
--nproc_per_node=1 \
tools/train.py \
--dataset_type=phonics_encode \
--encoder_type=ResNetGradualStyleEncoder \
--exp_dir=/content/drive/MyDrive/AGE/snakch/run1_1 \
--workers=8 \
--batch_size=8 \
--valid_batch_size=8 \
--valid_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--start_from_latent_avg \
--l2_lambda=1 \
--sparse_lambda=0.005 \
--orthogonal_lambda=0.0005 \
--A_length=100 \
--output_size=512 \
--class_embedding_path=/content/classs/embeddings/class_embeddings.pt \
--psp_checkpoint_path=/path/to/pretrained/pSp/checkpoints/iteration_300000.pt 
```
