#!/usr/bin/env python
import argparse
import os
import cv2
import numpy as np
import torch


N_IMAGES = 202599
ATTR_PATH = 'attributes.pth'


def preprocess_images(IMG_SIZE):
    IMG_PATH = 'images_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)

    if os.path.isfile(IMG_PATH):
        print("%s exists, nothing to do." % IMG_PATH)
        return

    print("Reading images from img_align_celeba/ ...")
    # allocate all memory at once
    data_torch=torch.ByteTensor(N_IMAGES,3,IMG_SIZE,IMG_SIZE)
    for i in range(1, N_IMAGES + 1):
        if i % 10000 == 0:
            print(i)
        image=cv2.imread('img_align_celeba/%06i.jpg' % i)[20:-20]
        assert image.shape == (178, 178, 3)
        if IMG_SIZE < 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        elif IMG_SIZE > 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        data_torch[i-1]=torch.from_numpy(image.transpose((2, 0, 1)))

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data_torch[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data_torch, IMG_PATH)


def preprocess_attributes():

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    attr_lines = [line.rstrip() for line in open('list_attr_celeba.txt', 'r')]
    assert len(attr_lines) == N_IMAGES + 2

    attr_keys = attr_lines[1].split()
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

    for i, line in enumerate(attr_lines[2:]):
        image_id = i + 1
        split = line.split()
        assert len(split) == 41
        assert split[0] == ('%06i.jpg' % image_id)
        assert all(x in ['-1', '1'] for x in split[1:])
        for j, value in enumerate(split[1:]):
            attributes[attr_keys[j]][i] = value == '1'

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)



parser = argparse.ArgumentParser(description='Preprocess images and convert them to torch file format')
parser.add_argument("--img_sz", type=int, default=256,
                    help="Image sizes (images have to be squared)")

params = parser.parse_args()


print("Preprocessing images and resizing to %ix%i"%(params.img_sz,params.img_sz))

preprocess_images(params.img_sz)
preprocess_attributes()
