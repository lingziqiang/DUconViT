# DUconViT
This is the code of the project DUconViT, which you can use to train your data. In addition, this code has codes for many models: FCN, Unet, Unet++, MCTrans, MedT, Swin Unet, Transunet, AttUnet, UTnet.

1.If you use colab, you can configure the environment with the following command:

#all

!pip install einops

!pip install timm

!pip install tensorboardX

!pip install yacs

!pip install medpy

!pip install -U PyYAML


  
#Transunet

!pip install ml_collections



#MCTrans

!pip install -r ./networks/mctrans/requirements.txt

!pip install monai

!pip install mmcv

import os

path="./networks/mctrans/models/ops"

os.chdir(path)

!python setup.py build install


Note: 

1)If you don't train mctrans, you can remove the reference to mctrans.

2)If you encounter problems configuring the environment for a model, you can go to the original paper on that model to find out how to configure the environment.



2.You can make a dataset in the same format as described in the data/readme.md.


3.train of test the model

python train.py --dataset dataset_name --max_epochs xxx --num_classes 2 ^

--root_path ./data/train_npz  --valid_dataset_path ../0.data/npz_h5_data/test_vol_h5 ^

--list_dir ./lists/lists_name ^

--output_dir ./results  --img_size 224 --base_lr 0.05 --batch_size 24 --n_gpu 1 --model  model_name


If you are interrupted during training and need to restart, add the following parameters:

--checkpoint_path results/params.pth --start_epoch xx --max_iou xx

