import os
import numpy as np

import argparse
import random
import tifffile
import csv

import torch
from torch.utils.data import Dataset

from feature_detectors import get_cloud_cloudshadow_mask

class AlignedDataset(Dataset):

    def __init__(self, opts, filelist):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opts = opts

        self.filelist = filelist
        self.n_images = len(self.filelist)

        self.clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                    [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = 1
        self.scale = 10000


    def __getitem__(self, index):

        fileID = self.filelist[index]

        # fileID format: [split_id, dataset_type, s1_folder, s2_folder, s2_cloudy_folder, s2_filename, s1_filename, s2_cloudy_filename]
        # For backward compatibility, check number of elements
        if len(fileID) == 5:
            # Old format: same filename for all
            s1_path = os.path.join(self.opts.input_data_folder, fileID[1], fileID[4])
            s2_cloudfree_path = os.path.join(self.opts.input_data_folder, fileID[2], fileID[4])
            s2_cloudy_path = os.path.join(self.opts.input_data_folder, fileID[3], fileID[4])
            reference_filename = fileID[4]
        elif len(fileID) == 7:
            # New format without dataset_type (winter only): [split_id, s1_folder, s2_folder, s2_cloudy_folder, s2_filename, s1_filename, s2_cloudy_filename]
            s1_path = os.path.join(self.opts.input_data_folder, fileID[1], fileID[5])
            s2_cloudfree_path = os.path.join(self.opts.input_data_folder, fileID[2], fileID[4])
            s2_cloudy_path = os.path.join(self.opts.input_data_folder, fileID[3], fileID[6])
            reference_filename = fileID[4]
        else:
            # New format with dataset_type: [split_id, dataset_type, s1_folder, s2_folder, s2_cloudy_folder, s2_filename, s1_filename, s2_cloudy_filename]
            dataset_type = fileID[1]
            
            # Construct correct root based on dataset type
            if dataset_type == 'winter':
                root = '/kaggle/input/sen12ms-cr-winter'
            elif dataset_type == 'spring':
                root = '/kaggle/input/t-glf-cr-winter'
            elif dataset_type == 'fall':
                root = '/kaggle/input/t-glf-cr-fall'
            else:
                root = self.opts.input_data_folder  # Fallback
            
            s1_path = os.path.join(root, fileID[2], fileID[6])
            s2_cloudfree_path = os.path.join(root, fileID[3], fileID[5])
            s2_cloudy_path = os.path.join(root, fileID[4], fileID[7])
            reference_filename = fileID[5]
            
        s1_data = self.get_sar_image(s1_path).astype('float32')
        s2_cloudfree_data = self.get_opt_image(s2_cloudfree_path).astype('float32')
        s2_cloudy_data = self.get_opt_image(s2_cloudy_path).astype('float32')

        if self.opts.is_use_cloudmask:
            cloud_mask = get_cloud_cloudshadow_mask(s2_cloudy_data, self.opts.cloud_threshold)
            cloud_mask[cloud_mask != 0] = 1
        '''
        for SAR, clip param: [-25.0, -32.5], [0, 0]
                 minus the lower boundary to be converted to positive
                 normalized by clip_max - clip_min, and increase by max_val
        for optical, clip param: 0, 10000
                     normalized by scale
        '''
        s1_data = self.get_normalized_data(s1_data, data_type=1)
        s2_cloudfree_data = self.get_normalized_data(s2_cloudfree_data, data_type=2)
        s2_cloudy_data = self.get_normalized_data(s2_cloudy_data, data_type=3)

        s1_data = torch.from_numpy(s1_data)
        s2_cloudfree_data = torch.from_numpy(s2_cloudfree_data)
        s2_cloudy_data = torch.from_numpy(s2_cloudy_data)
        if self.opts.is_use_cloudmask:
            cloud_mask = torch.from_numpy(cloud_mask)

        if self.opts.load_size - self.opts.crop_size > 0:
            if not self.opts.is_test:
                y = random.randint(0, np.maximum(0, self.opts.load_size - self.opts.crop_size))
                x = random.randint(0, np.maximum(0, self.opts.load_size - self.opts.crop_size))
            else:
                y = np.maximum(0, self.opts.load_size - self.opts.crop_size)//2
                x = np.maximum(0, self.opts.load_size - self.opts.crop_size)//2
            s1_data = s1_data[...,y:y+self.opts.crop_size,x:x+self.opts.crop_size]
            s2_cloudfree_data = s2_cloudfree_data[...,y:y+self.opts.crop_size,x:x+self.opts.crop_size]
            s2_cloudy_data = s2_cloudy_data[...,y:y+self.opts.crop_size,x:x+self.opts.crop_size]
            if self.opts.is_use_cloudmask:
                cloud_mask = cloud_mask[y:y+self.opts.crop_size,x:x+self.opts.crop_size]
        results = {'cloudy_data': s2_cloudy_data,
                   'cloudfree_data': s2_cloudfree_data,
                   'SAR_data': s1_data,
                   'file_name': reference_filename}
        if self.opts.is_use_cloudmask:
            results['cloud_mask'] = cloud_mask

        return results

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_images

    def get_opt_image(self, path):
        # Read TIFF file with tifffile (Kaggle compatible)
        image = tifffile.imread(path)
        # Ensure proper shape: (channels, height, width)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            # tifffile often returns (H, W, C). Detect channel-last where the last dim is small (<=20)
            h, w, c = image.shape
            if c <= 20 and h > c and w > c:
                image = np.transpose(image, (2, 0, 1))
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts
        return image

    def get_sar_image(self, path):
        # Read TIFF file with tifffile (Kaggle compatible)
        image = tifffile.imread(path)
        # Ensure proper shape: (channels, height, width)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            # tifffile often returns (H, W, C). Detect channel-last where last dim is small (<=20)
            h, w, c = image.shape
            if c <= 20 and h > c and w > c:
                image = np.transpose(image, (2, 0, 1))
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts
        return image

    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
            data_image /= self.scale

        return data_image
'''
read data.csv
'''
def get_train_val_test_filelists(listpath):

    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)

    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)

    csv_file.close()

    return train_filelist, val_filelist, test_filelist

if __name__ == "__main__":
    ##===================================================##
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='../data')
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=True)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    opts = parser.parse_args() 

    ##===================================================##
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ##===================================================##
    train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    ##===================================================##
    data = AlignedDataset(opts, test_filelist)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=False)

    ##===================================================##
    _iter = 0
    for results in dataloader:
        cloudy_data = results['cloudy_data']
        cloudfree_data = results['cloudfree_data']
        SAR = results['SAR_data']
        if opts.is_use_cloudmask:
            cloud_mask = results['cloud_mask']
        file_name = results['file_name']
        print(_iter, file_name)
        print('cloudy:', cloudy_data.shape)
        print('cloudfree:', cloudfree_data.shape)
        print('sar:', SAR.shape)
        if opts.is_use_cloudmask:
            print('cloud_mask:', cloud_mask.shape)
        _iter += 1
