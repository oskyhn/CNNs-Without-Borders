In this part, we give the rook experiment with [notebook](chess.ipynb) and implementation of experiments with Bagnet-33, Resnet-18 and Densenet-121 on 4-Quadrant Imagenet (4QI) dataset.

## 4-Quadrant Imagenet Experiment

### Requirements

 1. You need to have Imagenet 2012 validation set to generate the 4QI dataset.
 2. For Bagnet model, you need to have Bagnet implementation and the pretrained weights on Imagenet. Please check [Bagnet repository](https://github.com/wielandbrendel/bag-of-local-features-models).
 3. pytorch >= 1.2.0

### Usage

```
usage: QI_location.py [--arch] [--init_type][--epochs][--batch_size]  
               [--lr] [--momentum] [--weight_decay] [--nesterov]
               [--lr_min][--patience]

optional arguments:
  --arch                model architecture: resnet18 | bagnet33 |
                        | densenet121 (default: resnet18)
  --init_type           pretrained | stratch | random (default: pretrained)
  --batch-_size         mini-batch size (default: 8)
  --epochs              number of total epochs to run (default: 50)
  --lr LR,              initial learning rate (default: 1e-3)
  --momentum M          momentum (default: 0.9)
  --weight_decay        weight decay (default: 5e-5)
  --nesterov            nesterov (default: True)
  --lr_min              minimum learning rate (default: 1e-6)
  --patience            patience for reduce learning rate (default: 5)
```
