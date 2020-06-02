# This code is built from Kensho Hara's repository: https://github.com/kenshohara/3D-ResNets-PyTorch.
# Copyright (c) 2017 Kensho Hara.
# The repository is available under the MIT License.
#
# ==========================================================================================
# This implementation includes Same-Conv and Full-Conv versions all together. 
# For more details see the paper: 
# O. S. Kayhan and J. van Gemert,
# "On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location"
# In CVPR, 2020. 
# https://arxiv.org/abs/2003.07064
# ==========================================================================================
import torch
from torch import nn

from models import resnet_fconv_tem, resnet_fconv, resnet, pre_act_resnet, wide_resnet, resnext, densenet


def generate_model(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.f_conv:  # using Full-Conv

            if opt.temporal:  # with temporal Full-Conv
                print("Full-Conv-Temporal")
                from models.resnet_fconv_tem import get_fine_tuning_parameters

                if opt.model_depth == 10:
                    model = resnet_fconv_tem.resnet10(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 18:
                    model = resnet_fconv_tem.resnet18(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 34:
                    model = resnet_fconv_tem.resnet34(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 50:
                    model = resnet_fconv_tem.resnet50(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 101:
                    model = resnet_fconv_tem.resnet101(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 152:
                    model = resnet_fconv_tem.resnet152(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 200:
                    model = resnet_fconv_tem.resnet200(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)

            else:  # spatial Full-Conv

                from models.resnet_fconv import get_fine_tuning_parameters

                if opt.model_depth == 10:
                    model = resnet_fconv.resnet10(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 18:
                    model = resnet_fconv.resnet18(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 34:
                    model = resnet_fconv.resnet34(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 50:
                    model = resnet_fconv.resnet50(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 101:
                    model = resnet_fconv.resnet101(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 152:
                    model = resnet_fconv.resnet152(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
                elif opt.model_depth == 200:
                    model = resnet_fconv.resnet200(
                            num_classes=opt.n_classes,
                            shortcut_type=opt.resnet_shortcut,
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)

        else:  # using same convolution
            print("Same-Conv")
            from models.resnet import get_fine_tuning_parameters

            if opt.model_depth == 10:
                model = resnet.resnet10(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
            elif opt.model_depth == 18:
                model = resnet.resnet18(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
            elif opt.model_depth == 34:
                model = resnet.resnet34(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
            elif opt.model_depth == 50:
                model = resnet.resnet50(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
            elif opt.model_depth == 101:
                model = resnet.resnet101(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
            elif opt.model_depth == 152:
                model = resnet.resnet152(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
            elif opt.model_depth == 200:
                model = resnet.resnet200(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
           
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        from models.wide_resnet import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        from models.pre_act_resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        from models.densenet import get_fine_tuning_parameters

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.module.classifier = nn.Linear(
                    model.module.classifier.in_features, opt.n_finetune_classes)
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()
