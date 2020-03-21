import h5py
import numpy as np
import torch.utils.data as data
from PIL import Image


class Market(data.Dataset):
    def __init__(self, root, state='train', transform=None, require_views=True):
        #../data/Market.mat
        super(Market, self).__init__()
        self.root = root
        self.state = state
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False

        f = h5py.File(self.root, 'r')
        variables = list(f.items())
        '''variables
        [('gallery_data', <HDF5 dataset "gallery_data": shape (15913, 3, 128, 384), type "|u1">),
        ('gallery_labels', <HDF5 dataset "gallery_labels": shape (15913, 1), type "<i8">),
        ('gallery_views', <HDF5 dataset "gallery_views": shape (15913, 1), type "<i8">),
        ('probe_data', <HDF5 dataset "probe_data": shape (3368, 3, 128, 384), type "|u1">),
        ('probe_labels', <HDF5 dataset "probe_labels": shape (3368, 1), type "<i8">),
        ('probe_views', <HDF5 dataset "probe_views": shape (3368, 1), type "<i8">),
        ('train_data', <HDF5 dataset "train_data": shape (12936, 3, 128, 384), type "|u1">),
        ('train_labels', <HDF5 dataset "train_labels": shape (12936, 1), type "<i8">),
        ('train_views', <HDF5 dataset "train_views": shape (12936, 1), type "<i8">)]
        '''
        # [0]: gallery_data
        # [1]: gallery_labels
        # [2]: gallery_views
        # [3]: probe_data
        # [4]: probe_labels
        # [5]: probe_views
        # [6]: train_data
        # [7]: train_labels
        # [8]: train_views

        if self.state == 'train':
            _, temp = variables[6]
            self.data = np.transpose(temp[()], (0, 3, 2, 1))
            _, temp = variables[7]
            self.labels = np.squeeze(temp[()])
            _, temp = variables[8]
            self.views = np.squeeze(temp[()])
        elif self.state == 'gallery':
            _, temp = variables[0]
            self.data = np.transpose(temp[()], (0, 3, 2, 1))
            _, temp = variables[1]
            self.labels = np.squeeze(temp[()])
            _, temp = variables[2]
            self.views = np.squeeze(temp[()])
        elif self.state == 'probe':
            _, temp = variables[3]
            self.data = np.transpose(temp[()], (0, 3, 2, 1))
            _, temp = variables[4]
            self.labels = np.squeeze(temp[()])
            _, temp = variables[5]
            self.views = np.squeeze(temp[()])
        else:
            assert False, 'Unknown state: {}\n'.format(self.state)

    def return_mean(self, axis=(0, 1, 2)):
        return np.mean(self.data, axis)

    def return_std(self, axis=(0, 1, 2)):
        return np.std(self.data, axis)

    def return_num_class(self):
        return np.size(np.unique(self.labels))

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img, label, view = self.data[index], self.labels[index], self.views[index]

        img = Image.fromarray(img)

        if self.on_transform:
            img = self.transform(img)

        if self.require_views:
            return img, label, view, index
        else:
            return img, label


class FullTraining(data.Dataset):
    def __init__(self, root, transform=None, require_views=False):
        super(FullTraining, self).__init__()
        self.root = root #../data/MSMT17.mat
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False

        f = h5py.File(self.root, 'r') #<HDF5 file "MSMT17.mat" (mode r)>
        variables = list(f.items())
        #[('data', <HDF5 dataset "data": shape (124068, 3, 128, 384), type "|u1">), ('labels', <HDF5 dataset "labels": shape (124068, 1), type "<i8">)],124068个人的图片
        # [0]: data
        # [1]: labels

        _, temp = variables[0]
        self.data = np.transpose(temp[()], (0, 3, 2, 1)) #(124068, 384, 128, 3)
        _, temp = variables[1]
        self.labels = np.squeeze(temp[()])

    def return_mean(self, axis=(0, 1, 2)):
        if 'MSMT17' in self.root:
            return np.array([79.2386, 73.9793, 77.2493])
        else:
            return np.std(self.data, axis)

    def return_std(self, axis=(0, 1, 2)):
        if 'MSMT17' in self.root:
            return np.array([67.2012, 63.9191, 61.8367])
        else:
            return np.std(self.data, axis)

    def return_num_class(self):
        #4101个人id
        return np.size(np.unique(self.labels))

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        img = Image.fromarray(img)

        if self.on_transform:
            img = self.transform(img)

        return img, label


def main():
    pass


if __name__ == '__main__':
    main()
