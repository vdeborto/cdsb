import argparse
import torch
import numpy as np
import lmdb
import os

import sys
from tfrecord.torch.dataset import TFRecordDataset

def main(split, tfrecord_path, lmdb_path):
    assert split in {'train', 'validation'}
    num_images = 70000
    num_train = 63000

    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    file_ind = np.arange(num_train) if split == 'train' else np.arange(num_train, num_images)
    lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % split)

    print(file_ind)

    index_path = None
    description = {'shape': 'int', 'data': 'byte'}
    dataset = TFRecordDataset(tfrecord_path, index_path, description)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=1e12)
    i = 0
    count = 0
    txn = env.begin(write=True)
    for data in loader:
        if i in file_ind:
            im = data['data'][0].cpu().numpy()
            shape = data['shape'].flatten()
            im = np.reshape(im, [*shape]).transpose((1, 2, 0)).reshape((256, 256, 3))

            txn.put(str(count).encode(), np.ascontiguousarray(im))
            count += 1

            if count % 100 == 0:
                print(count)
                sys.stdout.flush()

            if count % 5000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        i = i + 1

    txn.commit()
    env.close()

    print('added %d items to the LMDB dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LMDB creator. Download images1024x1024.zip from here and unzip it \n'
                                     'https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS')
    # experimental results
    parser.add_argument('--tfrecord_path', type=str, default='data/ffhq/ffhq-r08.tfrecords',
                        help='location of tfrecord')
    parser.add_argument('--lmdb_path', type=str, default='data/ffhq/images256x256-lmdb',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=['train', 'validation'])
    args = parser.parse_args()

    main(args.split, args.tfrecord_path, args.lmdb_path)

