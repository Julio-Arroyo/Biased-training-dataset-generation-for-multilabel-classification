import os
import json
import numpy as np
import argparse

pp = argparse.ArgumentParser(description='Format PASCAL 2012 metadata.')
pp.add_argument('--load-path', type=str, default='/home/julioarroyo/research Eli and Julio/single-positive-multi-label-julio/data/pascal/', help='Path to a directory containing a copy of the PASCAL dataset.')
pp.add_argument('--save-path', type=str, default='/home/julioarroyo/research Eli and Julio/single-positive-multi-label-julio/data/pascal/', help='Path to output directory.')
args = pp.parse_args()

catName_to_catID = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
}

catID_to_catName = {catName_to_catID[k]: k for k in catName_to_catID}

ann_dict = {}
image_list = {'train': [], 'val': []}

for phase in ['train', 'val']:
    for cat in catName_to_catID:
        # this file tells you in what images does cat occurr
        with open(os.path.join(args.load_path, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', cat + '_' + phase + '.txt'), 'r') as f:
            for line in f:
                cur_line = line.rstrip().split(' ')
                image_id = cur_line[0]
                label = cur_line[-1]
                image_fname = image_id + '.jpg'
                # if cat occurs, save it in ann_dict, which maps from image to catID that occur in the image
                # image_list holds all images with annotated objects in them, classified by train or val
                if int(label) == 1:
                    if image_fname not in ann_dict:
                        ann_dict[image_fname] = []
                        image_list[phase].append(image_fname)
                    ann_dict[image_fname].append(catName_to_catID[cat])
    # create label matrix: 
    image_list[phase].sort()
    num_images = len(image_list[phase])
    label_matrix = np.zeros((num_images, len(catName_to_catID)))
    for i in range(num_images):
        cur_image = image_list[phase][i]
        label_indices = np.array(ann_dict[cur_image])
        label_matrix[i, label_indices] = 1.0 # puts a 1.0 at (i, j) where i is the i-th image and j is the index of the category
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels.npy'), label_matrix)
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images.npy'), np.array(image_list[phase]))
