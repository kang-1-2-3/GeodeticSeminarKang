import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import os
import cv2
import itertools
import re
# def load_image(filename):
#     ext = splitext(filename)[1]
#     if ext == '.npy':
#         return Image.fromarray(np.load(filename))
#     elif ext in ['.pt', '.pth']:
#         return Image.fromarray(torch.load(filename).numpy())
#     else:
#         return Image.open(filename)


# def unique_mask_values(idx, mask_dir, mask_suffix):
#     # print(idx)
#     print(re.search(idx[0] + '-' + idx[1], mask_dir))
#     mask_file = list(glob.glob(mask_dir, idx[0] + '-' + idx[1] + mask_suffix + '.*'))[0]
#     mask = np.asarray(cv2.imread(mask_file))
#     if mask.ndim == 2:
#         return np.unique(mask)
#     elif mask.ndim == 3:
#         mask = mask.reshape(-1, mask.shape[-1])
#         return np.unique(mask, axis=0)
#     else:
#         raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = glob.glob(images_dir,recursive=True)
        self.mask_dir = glob.glob(mask_dir,recursive=True)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        # print([os.path.split(file)[1].split('-')[1] for file in images_dir])
        self.ids_1 = [os.path.split(file)[1].split('-')[1] for file in self.images_dir]
        self.ids_2 = [os.path.split(file)[1].split('-')[2] for file in self.images_dir]
        # if not self.ids:
        #     raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids_1)} examples')
        logging.info('Scanning mask files to determine unique values')
        # idx = list(itertools.product(self.ids_1, self.ids_2))
        idx = list(zip(self.ids_1 ,self.ids_2))
        self.ids = idx
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), idx),
        #         total=len(self.ids_1))
        #     )

        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h, _ = pil_img.shape
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((w, h), dtype=np.int64)
            # for i, v in enumerate(mask_values):
            #     if img.ndim == 2:
            #         mask[img == v] = i
            #     else:
            #         mask[(img == v).all(-1)] = i
            mask[(img != 0).all(-1)] = 255
            mask = (255-mask)/255
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        # mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # img_file = list(self.images_dir.glob(name + '.*'))
        mask_file = f'/home/data/bbd1k_/patch-{name[0]}-{name[1]}-osm.png'
        img_file = f'/home/data/bbd1k_/patch-{name[0]}-{name[1]}-image.png'
        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = np.array(cv2.imread(mask_file))
        img = np.array(cv2.imread(img_file))
        # print(mask.shape, img.shape)
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        
        step=1
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
