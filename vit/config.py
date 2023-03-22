import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
# TODO: Modify batch size (default 512)
parser.add_argument('--batch_size', type=int, default=32)
# TODO: Modify lr(default CIFAR-10)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--lrf', type=float, default=0.01)
parser.add_argument('--in_channels', type=int, default=3)

parser.add_argument('--dataset_train_dir', type=str,
                    default="./data/CIFAR10",
                    help='The directory containing the train data.')
parser.add_argument('--dataset_test_dir', type=str,
                    default="./data/CIFAR10",
                    help='The directory containing the test data.')

parser.add_argument('--summary_dir', type=str,
                    default="./summary/vit_base_patch16_224_cifar10",
                    help='The directory of saving weights and tensorboard.')
parser.add_argument('--weights', type=str,
                    default="",
                    help='Initial weights path.')
parser.add_argument('--gpu', type=str, default='0,1,2',
                    help='Select gpu device.')

parser.add_argument('--model', type=str, default='vit_base_patch16_224_cifar10',
                    help='The name of ViT model. Select one to train.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
