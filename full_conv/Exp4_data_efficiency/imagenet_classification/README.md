# Image Classification with Full-Conv
We use Imagenet 2012 dataset and 4 random subsets derived from it and train different architectures. This code is built from the PyTorch examples repository: https://github.com/pytorch/examples/.


## 1. Same-Convolution:

Resnet and VGG architectures with Same-Convolution.

## Usage
```
usage: main.py [-h] [--data][--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [-t] [-sub] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               DIR

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  --data                path to dataset
  --arch ARCH, -a ARCH  model architecture: 
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | vgg11 | vgg11_bn | vgg13 
                        | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --num_class           number of classes (default: 1000)
```
Single node, multi-gpu training:

`python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data <imagenet-folder with train and val folders>`


## 2. Full-Convolution:

This implementation includes pytorch Resnet and VGG architectures with Full-Convolution.

The usage of Full-Conv is similar as Same-Conv. You only need to change the architecture name.

`python main.py -a resnet50_fconv --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data <imagenet-folder with train and val folders>`


## Imagenet-50 Subset Generation

We give the mapping of Imagenet-50 for reproducubility aspects. To generate Imagenet-50 dataset, first you need to download ImageNet dataset (2012). After that, you need to run the python script below: 

`python3 generate_images.py <imagenet_train_directory_path> <imagenet_50_directory_path>`

Example:

`python3 generate_images.py "/home/user/imagenet/train" "/home/user/imagenet50/"`


## Pretrained weights

### Full Imagenet:

[Resnet-50 with Same Conv](https://drive.google.com/file/d/1WhZJwlbLK-9uTwg3wcLzlSajcokTgCOv/view?usp=sharing)

[Resnet-50 with Full Conv](https://drive.google.com/file/d/1wGVdqNQfmd-_mUU5CEpbq-KkDiCriAfI/view?usp=sharing)

### Imagenet 50:

[Resnet-50 with Same Conv](https://drive.google.com/file/d/1EgYFbyJqCmr1uwx0EKW7zKd8pBIBNxcw/view?usp=sharing)

[Resnet-50 with Full Conv](https://drive.google.com/file/d/1VqlV10kFDJsSwos1B7HsbHLSjw50GuGq/view?usp=sharing)


## Evaluation

Same-Conv:

`python main.py -a resnet50 -e --resume resnet50_baseline_model_best.pth.tar --data <user/imagenet>`

Full-Conv:

`python main.py -a resnet50_fconv -e --resume resnet50_fconv_model_best.pth.tar --data <user/imagenet>`

