import os
import shutil

from torch import nn

from model_base import (vit_base_patch16_224_cifar10,
                        vit_base_patch7_28_mnist,
                        vit_base_patch16_224_cifar100)

def create_model(args):

    if args.model == 'vit_base_patch16_224_cifar10':
        model = vit_base_patch16_224_cifar10(args.num_classes, args.in_channels, has_logits=True)
    elif args.model == 'vit_base_patch16_224_cifar100':
        model = vit_base_patch16_224_cifar100(args.num_classes, args.in_channels, has_logits=True)
    elif args.model == 'vit_base_patch7_28_mnist':
        model = vit_base_patch7_28_mnist(args.num_classes, args.in_channels, has_logits=True)
    else:
        raise Exception("Error: Can't find any model name called {}.".format(args.model))

    return model

def model_parallel(args, model):

    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model

def remove_dir_and_create_dir(dir_name):
    """
    Remove existing directory, and create a new one.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Create OK.")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Create OK.")