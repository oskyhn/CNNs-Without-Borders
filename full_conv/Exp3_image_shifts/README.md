# Image Classification, Diagonal Shift and Consistency with Full-Conv

We choose 200-class [Imagenet-A dataset](https://arxiv.org/abs/1907.07174) for this experiment and train different architectures. This code is built from the PyTorch examples repository: https://github.com/pytorch/examples/ and Adobe's repository: https://github.com/adobe/antialiased-cnns.

You can find 4 different methods for each architecture: Same-Conv, Full-Conv, Same-Conv with [BlurPool](https://github.com/adobe/antialiased-cnns) and Full-Conv with BlurPool.

## Training

### 1. Same-Convolution:

`python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data <imagenet-folder with train and val folders>`

### 2. Full-Convolution:

`python main.py -a resnet50_fconv --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data <imagenet-folder with train and val folders>`

### 3. Same-Convolution with BlurPool:

`python main.py -a resnet50_lpf3 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data <imagenet-folder with train and val folders>`

### 4. Full-Convolution with Blurpool:

`python main.py -a resnet50_fconv_lpf3 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data [imagenet-folder with train and val folders]`

**Note:** You can change the blurpool filter size by changing number next to **lpf** argument. (Supported filter sizes: [1,7])

## Usage

```
usage: main.py [-h] [--data][--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  --data                path to dataset 
  --arch ARCH, -a ARCH  model architecture: resnet101 | resnet152 |
                        resnet18 | resnet34 |resnet50 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
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
  --num_class           number of classes (default: 200)
  --shift_inc           increasing the shifting (default: 32)
  --no-data-aug         no shift-based data augmentation
  --out-dir             output directory
  -es, --evaluate-shift evaluate model on shift-invariance
  --epochs-shift        number of total epochs to run for shift-invariance test
                        (default: 5)
  -ed, --evaluate-diagonal
                        evaluate model on diagonal
  -ba, --batch-accum    number of mini-batches to accumulate gradient over
                        before updating(default: 1)                     
```
