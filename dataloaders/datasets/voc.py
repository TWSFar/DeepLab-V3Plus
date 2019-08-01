import os
import os.path as osp
import numpy as np
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
try:
    from dataloaders import custom_transforms as ctf
    from mypath import Path
except:
    import sys
    sys.path.extend(['C:\\Users\\TWSF\\Desktop\\ResNet\\dataloaders', 
                     'C:\\Users\\TWSF\\Desktop\\ResNet\\', 
                     'C:\\Users\\TWSF\\Desktop\\ResNet\\dataloaders\\datasets'])
    from dataloaders import custom_transforms as ctf
    from mypath import Path

    print('test back door')


class VOCSegmentation(Dataset):

    NUM_CLASSES = 21
    
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('voc'),
                 split='train'):
        
        super().__init__()
        self.args = args
        self._base_dir = base_dir
        self._img_dir = osp.join(self._base_dir, 'JPEGImages')
        self._cat_dir = osp.join(self._base_dir, 'SegmentationClass')
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        
        for splt in self.split:
            with open(osp.join(_splits_dir, split + '.txt'), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = glob(osp.join(self._img_dir, line + '*'))
                _cat = glob(osp.join(self._cat_dir, line + '*'))
                assert (osp.isfile(_image[0]) and len(_image) == 1)
                assert (osp.isfile(_cat[0]) and len(_cat) == 1)
                self.im_ids.append(line)
                self.images.append(_image[0])
                self.categories.append(_cat[0])

            assert (len(self.images) == len(self.categories))

            # Display stats
            print('Number of images in {}: {:d}'.format(split, len(self.images)))


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            return self._transforms(sample, split)
    
    def _transforms(self, sample, split):
        if split == 'train':
            composed_transforms = transforms.Compose([
                ctf.RandomHorizontalFlip(),
                ctf.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                ctf.RandomGaussianBlur(),
                ctf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ctf.ToTensor()])
            return composed_transforms(sample)

        elif split == 'val':
            composed_transforms = transforms.Compose([
                ctf.FixScaleCrop(crop_size=self.args.crop_size), # default = 513
                ctf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ctf.ToTensor()])
            return composed_transforms(sample)
    
    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return "VOC2012(split=" + str(self.split) + ')'


if __name__ == "__main__":
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='voc')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)