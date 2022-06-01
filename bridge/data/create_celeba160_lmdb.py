import argparse
import torch
import numpy as np
import lmdb
import os

from PIL import Image

import sys
sys.path.append('gitmodules/matlab_imresize')
from imresize import imresize

def main(split, img_path, lmdb_path):
    assert split in {'train', 'validation', 'test'}
    num_images = 202599
    num_train = 162770
    num_valid = 19867

    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    if split == 'train':
        file_ind = np.arange(num_train)
    elif split == 'validation':
        file_ind = np.arange(num_train, num_train + num_valid)
    else:
        file_ind = np.arange(num_train + num_valid, num_images)
    lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % split)

    print(file_ind)

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=1e12)
    count = 0
    txn = env.begin(write=True)
    for i in file_ind:
        im = Image.open(os.path.join(img_path, '%06d.png' % (i + 1)))
        # im = im.resize(size=(256, 256), resample=Image.BILINEAR)
        im = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[1], im.size[0], 3)

        im = imresize(im, output_shape=(160, 160))

        txn.put(str(count).encode(), np.ascontiguousarray(im))
        count += 1

        if count % 100 == 0:
            print(count)
            sys.stdout.flush()

        if count % 5000 == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    print('added %d items to the LMDB dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LMDB creator. Download images1024x1024.zip from here and unzip it \n'
                                     'https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS')
    # experimental results
    parser.add_argument('--img_path', type=str, default='data/celeba160/align_size(640,640)_move(0.250,0.000)_face_factor(0.600)_jpg/data',
                        help='location of images')
    parser.add_argument('--lmdb_path', type=str, default='data/celeba160/celeba-lmdb',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=['train', 'validation', 'test'])
    args = parser.parse_args()

    main(args.split, args.img_path, args.lmdb_path)

