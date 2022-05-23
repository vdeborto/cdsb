import argparse
import numpy as np
import torch
import os
from PIL import Image

from tfrecord.torch.dataset import TFRecordDataset


def main(dataset, split, tfr_path, out_path):
    assert split in {'train', 'validation'}

    # create target directory
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    if dataset == 'celeba' and split in {'train', 'validation'}:
        num_shards = {'train': 120, 'validation': 40}[split]
        out_path = os.path.join(out_path, split)
        tfrecord_path_template = os.path.join(tfr_path, '%s/%s-r08-s-%04d-of-%04d.tfrecords')
    elif dataset == 'imagenet-oord_32':
        num_shards = {'train': 2000, 'validation': 80}[split]
        # imagenet_oord_out_path += '_32'
        out_path = os.path.join(out_path, split)
        tfrecord_path_template = os.path.join(tfr_path, '%s/%s-r05-s-%04d-of-%04d.tfrecords')
    elif dataset == 'imagenet-oord_64':
        num_shards = {'train': 2000, 'validation': 80}[split]
        # imagenet_oord_out_path += '_64'
        out_path = os.path.join(out_path, split)
        tfrecord_path_template = os.path.join(tfr_path, '%s/%s-r06-s-%04d-of-%04d.tfrecords')
    else:
        raise NotImplementedError

    if os.path.exists(out_path):
        os.rmdir(out_path)
    os.mkdir(out_path)

    # create image folder
    count = 0
    for tf_ind in range(num_shards):
        # read tf_record
        tfrecord_path = tfrecord_path_template % (split, split, tf_ind, num_shards)
        index_path = None
        description = {'shape': 'int', 'data': 'byte', 'label': 'int'}
        dataset = TFRecordDataset(tfrecord_path, index_path, description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # put the data in lmdb
        for data in loader:
            im = data['data'][0].cpu().numpy()
            shape = data['shape'].flatten()
            im = np.reshape(im, [*shape])
            im = Image.fromarray(im, mode='RGB')
            im.save(os.path.join(out_path, str(count) + '.png'))
            count += 1
            if count % 100 == 0:
                print(count)

    print('added %d items to the folder dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Folder creator using TFRecords from GLOW.')
    # experimental results
    parser.add_argument('--dataset', type=str, default='imagenet-oord_32',
                        help='dataset name', choices=['imagenet-oord_32', 'imagenet-oord_32', 'celeba'])
    parser.add_argument('--tfr_path', type=str, default='/data1/datasets/imagenet-oord/mnt/host/imagenet-oord-tfr',
                        help='location of TFRecords')
    parser.add_argument('--out_path', type=str, default='/data1/datasets/imagenet-oord/imagenet-oord',
                        help='target location for storing output files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=['train', 'validation'])
    args = parser.parse_args()
    main(args.dataset, args.split, args.tfr_path, args.out_path)



