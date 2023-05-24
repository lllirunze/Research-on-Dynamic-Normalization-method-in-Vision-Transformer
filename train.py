from __future__ import print_function

import functools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import math

import os
import argparse
import shutil
from torch.utils.tensorboard import SummaryWriter
from vit import ViT
from vit_dtn import ViT_DTN
from vit_un import ViT_UN
from vit_bn import ViT_BN
from t2t_vit import T2T_ViT
from t2t_vit_dtn import T2T_ViT_DTN
from t2t_vit_un import T2T_ViT_UN
from t2t_vit_bn import T2T_ViT_BN
from UN.unified_normalization import UN1d
from autoaugment import CIFAR10Policy, ImageNetPolicy

parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1200)
# TODO: Modify batch size (default 512)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument('--lrf', type=float, default=1e-5)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--model', type=str, default='vit-b')
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--summary_dir', type=str,
                    default="./summary/vit_small_patch16_224_cifar10",
                    help='The directory of saving weights and tensorboard.')
parser.add_argument('--weights', type=str,
                    default="",
                    help='Initial weights path.')
parser.add_argument('--gpu', type=str, default='0,1,2',
                    help='Select gpu device.')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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


def model_parallel(args, model):
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)
    return model


def train(args):
    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    # Record Weights
    weights_dir = args.summary_dir + "/weights"
    remove_dir_and_create_dir(weights_dir)

    # Record Logs
    log_dir = args.summary_dir + "/logs"
    remove_dir_and_create_dir(log_dir)
    writer = SummaryWriter(log_dir)

    # Define the multiprocess
    number_of_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process.'.format(number_of_workers))

    if args.data == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data/CIFAR10',
                                         train=True,
                                         transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       CIFAR10Policy(),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                            [0.2023, 0.1994, 0.2010])]),
                                         download=True)
        test_dataset = datasets.CIFAR10(root='./data/CIFAR10',
                                        train=False,
                                        transform=transforms.Compose([transforms.Resize(256),
                                                                      transforms.CenterCrop(224),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                           [0.2023, 0.1994, 0.2010])]),
                                        download=True)
    elif args.data == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data/CIFAR100',
                                          train=True,
                                          transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        CIFAR10Policy(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                             [0.2023, 0.1994, 0.2010])]),
                                          download=True)
        test_dataset = datasets.CIFAR100(root='./data/CIFAR100',
                                         train=False,
                                         transform=transforms.Compose([transforms.Resize(256),
                                                                       transforms.CenterCrop(224),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                            [0.2023, 0.1994, 0.2010])]),
                                         download=True)
    elif args.data == 'imagenet1k':
        # ILSVRC2012 is available locally.
        train_dataset = datasets.ImageFolder(root='/home/sdc1/dataset/ILSVRC2012/images/train',
                                             transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                           transforms.RandomHorizontalFlip(),
                                                                           ImageNetPolicy(),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                                                [0.229, 0.224, 0.225])]))
        test_dataset = datasets.ImageFolder(root='/home/sdc1/dataset/ILSVRC2012/images/val',
                                            transform=transforms.Compose([transforms.Resize(256),
                                                                          transforms.CenterCrop(224),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                                               [0.229, 0.224, 0.225])]))

    else:
        raise Exception("Error: Can't find data directory name called {}.".format(args.data))
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=number_of_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=number_of_workers)
    number_of_train = len(train_dataset)
    number_of_test = len(test_dataset)
    print('Using {} images for training.'.format(number_of_train))
    print('Using {} images for testing.'.format(number_of_test))

    if args.model == 'vit-b':
        model = ViT(image_size=args.image_size,
                    patch_size=args.patch_size,
                    num_classes=args.num_classes,
                    dim=768,
                    depth=12,
                    heads=12,
                    mlp_dim=3072,
                    dropout=args.dropout)
    elif args.model == 'vit-s':
        model = ViT(image_size=args.image_size,
                    patch_size=args.patch_size,
                    num_classes=args.num_classes,
                    dim=384,
                    depth=12,
                    heads=6,
                    mlp_dim=1536,
                    dropout=args.dropout)
    elif args.model == 'vit-t':
        model = ViT(image_size=args.image_size,
                    patch_size=args.patch_size,
                    num_classes=args.num_classes,
                    dim=192,
                    depth=12,
                    heads=3,
                    mlp_dim=768,
                    dropout=args.dropout)
    elif args.model == 'vit-s-dtn':
        model = ViT_DTN(img_size=args.image_size,
                        patch_size=args.patch_size,
                        num_classes=args.num_classes,
                        embed_dim=384,
                        depth=12,
                        num_heads=6,
                        mlp_ratio=4.)
    elif args.model == 'vit-s-un':
        norm_layer_ = functools.partial(UN1d, window_size=4, warmup_iters=4000, outlier_filtration=True)
        model = ViT_UN(image_size=args.image_size,
                       patch_size=args.patch_size,
                       num_classes=args.num_classes,
                       dim=384,
                       depth=12,
                       heads=6,
                       mlp_dim=1536,
                       dropout=args.dropout,
                       norm_layer=norm_layer_)
    elif args.model == 'vit-s-bn':
        model = ViT_BN(image_size=args.image_size,
                       patch_size=args.patch_size,
                       num_classes=args.num_classes,
                       dim=384,
                       depth=12,
                       heads=6,
                       mlp_dim=1536,
                       dropout=args.dropout)
    elif args.model == 't2t-vit-s':
        model = T2T_ViT(image_size=args.image_size,
                        patch_size=args.patch_size,
                        num_classes=args.num_classes,
                        dim=384,
                        depth=12,
                        heads=6,
                        mlp_dim=1536,
                        dropout=args.dropout,
                        norm_layer=nn.LayerNorm)
    elif args.model == 't2t-vit-s-bn':
        model = T2T_ViT_BN(image_size=args.image_size,
                           patch_size=args.patch_size,
                           num_classes=args.num_classes,
                           dim=384,
                           depth=12,
                           heads=6,
                           mlp_dim=1536,
                           dropout=args.dropout,
                           norm_layer=nn.LayerNorm)
    elif args.model == 't2t-vit-s-un':
        norm_layer_ = functools.partial(UN1d, window_size=4, warmup_iters=4000, outlier_filtration=True)
        model = T2T_ViT_UN(image_size=args.image_size,
                           patch_size=args.patch_size,
                           num_classes=args.num_classes,
                           dim=384,
                           # depth=12,
                           depth=14,
                           heads=6,
                           # mlp_dim=1536,
                           mlp_dim=1152,
                           dropout=args.dropout,
                           norm_layer=norm_layer_)
    elif args.model == 't2t-vit-s-dtn':
        model = T2T_ViT_DTN(img_size=args.image_size,
                            patch_size=args.patch_size,
                            num_classes=args.num_classes,
                            embed_dim=384,
                            # depth=12,
                            depth=14,
                            num_heads=6,
                            mlp_ratio=4.,
                            drop_rate=0.1)
    else:
        raise Exception("Error: Can't find any model name called {}.".format(args.model))

    # If we have pre-trained weight and bias, we can just load them.
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        '''
        # Delete unnecessary weights
        del_keys = ['module.head.weight', 'module.head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'module.head.weight', 'module.head.bias']
        for k in del_keys:
            del weights_dict[k]
        '''
        # TODO: Delete 'module.'
        weights = {}
        for k, v in weights_dict.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights[new_k] = v
        print(model.load_state_dict(weights, strict=False))

    model = model_parallel(args, model)
    model.to(device)

    # Define loss function
    loss_function = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=5e-5)
    # TODO: Consine Annealing
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lrf)

    for epoch in range(args.epochs):
        # Train model
        model.train()
        # Define train accuracy and loss
        train_accuracy = 0.0
        train_loss = 0.0

        # Display a visible process bar
        train_bar = tqdm(train_loader)
        for data in train_bar:
            # Display train process
            train_bar.set_description("epoch {}".format(epoch))
            # Get images and labels
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # Empty the gradient
            optimizer.zero_grad()
            # Input the data into model
            outputs = model(images)
            prediction = (outputs.argmax(dim=1) == labels).float().mean()
            # Calculate the loss
            loss = loss_function(outputs, labels)
            # Get gradient automatically
            loss.backward()
            # TODO: Gradient Clip
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # Optimize the model parameters
            optimizer.step()
            scheduler.step()
            # Print statistics
            train_loss += (loss.item()) / len(train_loader)
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))
            train_accuracy += prediction / len(train_loader)
            # Clear batch variables from memory
            del images, labels

        # Test model every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            # Define test accuracy and loss
            test_accuracy = 0.0
            test_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    # Get images and labels
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # Input the data into model
                    outputs = model(images)
                    # Calculate the loss
                    loss = loss_function(outputs, labels)
                    prediction = (outputs.argmax(dim=1) == labels).float().mean()
                    # Print statistics
                    test_loss += (loss.item()) / len(test_loader)
                    test_accuracy += prediction / len(test_loader)
                    # Clear batch variables from memory
                    del images, labels

            # Statistics of data
            print("=> train_loss: {:.4f}, train_accuracy: {:.4f}, test_loss: {:.4f}, test_accuracy: {:.4f}".
                  format(train_loss, train_accuracy, test_loss, test_accuracy))

            # Record data by TensorBoard
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_accuracy, epoch)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        else:
            # Statistics of data
            print("=> train_loss: {:.4f}, train_accuracy: {:.4f}".format(train_loss, train_accuracy))

            # Record data by TensorBoard
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_accuracy, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                       "{}/epoch={}_train_acc={:.4f}_test_acc={:.4f}.pth".format(weights_dir, epoch + 1, train_accuracy,
                                                                                 test_accuracy))


if __name__ == '__main__':
    train(args)
