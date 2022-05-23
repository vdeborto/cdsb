import argparse
import torch
import numpy as np
import lmdb
import os

from PIL import Image


def main(split, ffhq_img_path, ffhq_lmdb_path):
    assert split in {'train', 'validation'}
    num_images = 70000
    num_train = 63000

    # create target directory
    if not os.path.exists(ffhq_lmdb_path):
        os.makedirs(ffhq_lmdb_path, exist_ok=True)

    file_ind = np.arange(num_train) if split == 'train' else np.arange(num_train, num_images)
    lmdb_path = os.path.join(ffhq_lmdb_path, '%s.lmdb' % split)

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=1e12)
    count = 0
    with env.begin(write=True) as txn:
        for i in file_ind:
            img_path = os.path.join(ffhq_img_path, '%05d.png' % i)
            im = Image.open(img_path)
            # im = im.resize(size=(256, 256), resample=Image.BILINEAR)
            im = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[1], im.size[0], 3)

            txn.put(str(count).encode(), im)
            count += 1
            if count % 100 == 0:
                print(count)

        print('added %d items to the LMDB dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FFHQ LMDB creator. Download images1024x1024.zip from here and unzip it \n'
                                     'https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS')
    # experimental results
    parser.add_argument('--ffhq_img_path', type=str, default='/data1/datasets/ffhq/images1024x1024',
                        help='location of images from FFHQ')
    parser.add_argument('--ffhq_lmdb_path', type=str, default='/data1/datasets/ffhq/ffhq-lmdb',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=['train', 'validation'])
    args = parser.parse_args()

    main(args.split, args.ffhq_img_path, args.ffhq_lmdb_path)

