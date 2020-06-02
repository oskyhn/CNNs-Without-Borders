In this part, we give implementation of 3D-Resnet with Full-Conv. 3D-Resnet implementation is built from [Kensho Hara's repository](https://github.com/kenshohara/3D-ResNets-PyTorch).

Please follow the steps in that repository for requirements and arranging the dataset.

## 3D-Resnet Full-Conv

We have 2 different implementations of Full-Conv:

**1. Full-Conv on spatial dimension.** 

`python main.py --root_path <your path> --video_path <your video path> --annotation_path <your annotation path/ucf101_01.json> --result_path <your path for results>  --dataset ucf101  --model resnet --model_depth 18 --n_classes 101 --batch_size 32 --n_threads 4 --checkpoint 5`

**2. Full-Conv on spatial and temporal dimension.** After the submission, we extended the Full-Conv on temporal dimension and the new model obtained an extra **1.6%** increase with 3D-Resnet18 arhitecture on UCF101 dataset.

`python main.py --root_path <your path> --video_path <your video path> --annotation_path <your annotation path/ucf101_01.json> --result_path <your path for results>  --dataset ucf101  --model resnet --model_depth 18 --n_classes 101 --batch_size 32 --n_threads 4 --checkpoint 5 --temporal True`

**Note:** If you want to use a network with Same-Conv, then use `--f_conv False` argument.


