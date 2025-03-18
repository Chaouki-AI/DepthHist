# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications
# by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# it contains the dataloader used to read, split, preprosess and build the dataloaders
# please Modify the args on the args_****.txt files to controle the Batch size and the number of images used for training 
# Contact: medchaoukiziara@gmail.com || chaouki.ziara@univ-sba.dz



import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])
class NoOpTransform:
    def __call__(self, x):
        return x

class DepthDataLoader(object):
    def __init__(self, args, images, depths, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, images, depths, mode, transform=preprocessing_transforms(mode))
            self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.bs,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=8,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, images, depths, mode, transform=preprocessing_transforms(mode))
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=8)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, images, depths, mode, transform=None, is_for_online_eval=False):
        self.args = args

        self.images = images
        self.depths = depths

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):

        if self.mode == 'train':
            
            image_path = self.images[idx]
            depth_path = self.depths[idx]
            mask = depth_path.replace('depth.npy', 'depth_mask.npy') if self.args.dataset == 'diode' else None

            image    = Image.open(image_path)
            depth_gt = np.load(depth_path) if self.args.dataset == 'diode' else Image.open(depth_path)
            mask_gt  = np.load(mask) if self.args.dataset == 'diode' else None #np.load(mask)
            if self.args.dataset == 'diode':
                depth_gt = (depth_gt*mask_gt[...,None]).squeeze()
                depth_gt = Image.fromarray(depth_gt)
            
            if self.args.dataset == 'kitti':
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))
            if random.random() > 0.5:
                random_angle = (random.random() - 0.5) * 2 * 2.5
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == 'nyu' or self.args.dataset == 'sunrgbd':
                depth_gt = depth_gt / 1000.0
            elif self.args.dataset == 'kitti':
                depth_gt = depth_gt / 256.0
            image, depth_gt = self.random_crop(image, depth_gt, self.args.image_height, self.args.image_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt}

        else:
            image_path = self.images[idx]
            image = Image.open(image_path)

            if self.mode == 'online_eval':
                depth_path = self.depths[idx]
                mask = depth_path.replace('depth.npy', 'depth_mask.npy') if self.args.dataset == 'diode' else None
                has_valid_depth = False
                try:
                    #depth_gt = Image.open(depth_path)
                    depth_gt = np.load(depth_path) if self.args.dataset == 'diode' else Image.open(depth_path)
                    mask_gt  = np.load(mask) if self.args.dataset == 'diode' else None
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                if has_valid_depth:
                    if self.args.dataset == 'diode':
                        depth_gt = (depth_gt*mask_gt[..., None]).squeeze()
                    depth_gt = np.asarray(depth_gt, dtype=np.float32) 
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    image    = np.asarray(image, dtype=np.float32) / 255.0
                    if self.args.dataset == 'nyu' or self.args.dataset == 'sunrgbd':
                        depth_gt = depth_gt / 1000.0
                    elif self.args.dataset == 'kitti':
                        depth_gt = depth_gt / 256.0

                
                if self.args.dataset == 'kitti':
                    height = image.shape[0]
                    width = image.shape[1]
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    if self.mode == 'online_eval' and has_valid_depth:
                        depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    

                if self.mode == 'online_eval':
                    sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.images)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #self.normalize  = transforms.Compose([NoOpTransform()])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth}
        else:
            if self.mode == 'visualize':
                return {'image': image, 'depth': depth}
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth,'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img