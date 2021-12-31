import os,sys

import argparse

parser = argparse.ArgumentParser(description='Download data.')
parser.add_argument('--data', type=str, help='mnist, celeba')
parser.add_argument('--data_dir', type=str, help='download location')
parser.add_argument('--task', type=str, help='Conditional task')


sys.path.append('..')


from bridge.data.stackedmnist import Stacked_MNIST, Cond_Stacked_MNIST
from bridge.data.emnist import EMNIST
from bridge.data.celeba import CelebA


# SETTING PARAMETERS

def main():

    args = parser.parse_args()

    if args.data == 'mnist':
        root = os.path.join(args.data_dir, 'mnist')
        Stacked_MNIST(root, source_root=root, train=True, num_channels=1, imageSize=28)
        Stacked_MNIST(root, source_root=root, train=False, num_channels=1, imageSize=28)

    if args.data == 'cond_mnist':
        root = os.path.join(args.data_dir, 'mnist')
        Cond_Stacked_MNIST(args, root=root, load=False, split='train', num_channels=1)
        Cond_Stacked_MNIST(args, root=root, load=False, split='valid', num_channels=1)
        Cond_Stacked_MNIST(args, root=root, load=False, split='test', num_channels=1)

    if args.data == 'celeba':
        root = os.path.join(args.data_dir, 'celeba')
        CelebA(root, split='all', download=True)
    

if __name__ == '__main__':
    main()  
