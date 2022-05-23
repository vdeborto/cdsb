import os,sys

import argparse

parser = argparse.ArgumentParser(description='Download data.')
parser.add_argument('--data', type=str, help='mnist, celeba')
parser.add_argument('--data_dir', type=str, help='download location')
parser.add_argument('--task', type=str, help='Conditional task')


sys.path.append('..')


from bridge.data.stackedmnist import Stacked_MNIST, Cond_Stacked_MNIST
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
        task = args.task
        Cond_Stacked_MNIST(task, root=root, load=False, split='train', num_channels=1)
        Cond_Stacked_MNIST(task, root=root, load=False, split='valid', num_channels=1)
        Cond_Stacked_MNIST(task, root=root, load=False, split='test', num_channels=1)

    if args.data == 'celeba':
        root = args.data_dir
        try:
            CelebA(root, split='all', download=True)

        except:
            import zipfile
            from torchvision.datasets.utils import download_file_from_google_drive
            import shutil

            data_dir = os.path.join(root, "celeba")

            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)

            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            download_file_from_google_drive("1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ", root, "celeba.zip", "70ca6a664804f6967e495e9e95904675")

            with zipfile.ZipFile(os.path.join(root, "celeba.zip"), "r") as f:
                f.extractall(root)

            CelebA(root, split='all', download=True)


if __name__ == '__main__':
    main()  
