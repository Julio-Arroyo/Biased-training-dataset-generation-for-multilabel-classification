import os
# import cv2
import json
import math
import random
import numpy as np
from format_pascal import catID_to_catName
from format_pascal import catName_to_catID
from format_coco import parse_categories

data_path = '/media/julioarroyo/aspen/data'
coco_path = '/coco/annotations/'
labels_path = '/home/julioarroyo/research_Eli_and_Julio/single-positive-multi-label-julio/data/coco'
SEED = 10

f_train = open(data_path + coco_path + 'instances_train2014.json')
f_val = open(data_path + coco_path + 'instances_val2014.json')
files = {'train' : json.load(f_train), 'val' : json.load(f_val)}


def get_anns_by_imID():
    imID_to_anns = {}
    for phase in ['train', 'val']:
        D = files[phase]
        for i in range(len(D['annotations'])):
            if not D['annotations'][i]['image_id'] in imID_to_anns:
                imID_to_anns[D['annotations'][i]['image_id']] = []
            imID_to_anns[D['annotations'][i]['image_id']].append(D['annotations'][i])
    return imID_to_anns


def get_sum_areas(annotations):
    sum_area = 0
    for ann in annotations:
        sum_area += ann['area']
    return sum_area


def make_biased_matrix(imID_to_anns, bias_type, num_categories, cat_id_to_index):
    for split in ['train', 'val']:
        D = files[split]
        image_id_and_weights = []
        image_id_list = sorted(np.unique([str(D['annotations'][i]['image_id']) for i in range(len(D['annotations']))]))
        image_id_list = np.array(image_id_list, dtype=int)
        image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}

        for im_id in image_id_list:
            weights_i = [0 for _ in range(num_categories)]
            if bias_type == 'size':
                sum_areas = get_sum_areas(imID_to_anns[im_id])
                assert sum_areas > 0
                annotations = imID_to_anns[im_id]
                for j in range(len(annotations)):
                    weight = annotations[j]['area'] / sum_areas
                    cat_id = annotations[j]['category_id']
                    cat_idx = cat_id_to_index[cat_id]
                    weights_i[cat_idx] += weight
            image_id_and_weights.append((im_id, weights_i))
        
        label_matrix = np.load(os.path.join(labels_path, 'formatted_{}_labels.npy'.format(split)))
        biased_matrix = np.zeros_like(label_matrix)
        for k in range(len(image_id_and_weights)):
            im_id = int(image_id_and_weights[k][0])
            weights = image_id_and_weights[k][1]
            row_idx = image_id_to_index[im_id]
            labels = label_matrix[row_idx][:]
            assert len(labels) == len(weights)
            col_idx = random.choices(range(len(labels)), weights=weights)[0]
            assert label_matrix[row_idx][col_idx] == 1
            biased_matrix[row_idx][col_idx] = 1
        np.save(os.path.join(labels_path,
                             'coco_formatted_{}_{}_{}_labels_obs.npy'.format(split, bias_type, 1)),
                             biased_matrix)



if __name__ == '__main__':
    imID_to_anns = get_anns_by_imID()
    (category_list, cat_id_to_index) = parse_categories(files['train']['categories'])
    bias_type = 'size'
    random.seed(SEED)
    make_biased_matrix(imID_to_anns, bias_type, len(category_list), cat_id_to_index)


# def get_imID_to_dims():
#     imID_to_dims = {}
#     for phase in ['train', 'phase']:
#         f = open(data_path + coco_path + 'instances_{}2014.json'.format(phase))
#         D = json.load(f)
#         for i in range(len(D['images'])):
#             imID_to_dims[D['images'][i]['id']] = [D['images'][i]['width'], D['images'][i]['height']]
#         f.close()
#     return imID_to_dims


# def get_center_bias_norm_const(annotations, imID_to_dims):
#     sum_inv_dist = 0
#     for ann in annotations:
#         center = imID_to_dims[ann['image_id']]
#         x_i = ann['bbox'][0] + ann['bbox'][2]/2 # x-coord. of center of annotation
#         y_i = ann['bbox'][1] + ann['bbox'][3]/2 # y-coord. of center of annotation
#         d_i = math.sqrt((x_i - center[0])**2 + (y_i - center[1])**2)
#         if d_i == 0:
#             d_i = 1
#         sum_inv_dist += 1/d_i
#     return 1/sum_inv_dist


# def get_weights_per_im(imID_to_anns, bias_type):
#     imID_to_catWeights = {}
#     imID_to_annWeights = {}
#     for imID in imID_to_anns:
#         # imID_to_catWeights[imID][i] will be the weight of the (i + 1)-th category as given by D['categories']
#         # reason i + 1 is that pascal_ann.json 1-indexed the categories
#         imID_to_catWeights[imID] = {meta['category_id_to_index'][cat_id] : 0 for cat_id in meta['category_id_to_index']}
#         if bias_type == 'size':
#             sum_areas = get_sum_areas(imID_to_anns[imID])
#             assert sum_areas > 0
#             for i in range(len(imID_to_anns[imID])):
#                 weight = imID_to_anns[imID][i]['area'] / sum_areas
#                 cat_col_idx = meta['category_id_to_index'][imID_to_anns[imID][i]['category_id']]
#                 imID_to_catWeights[imID][cat_col_idx] += weight
#         elif bias_type == 'location':
#             assert False  # TODO: Make location bias work too
#             imID_to_dims = get_imID_to_dims()
#             norm_const = get_center_bias_norm_const(imID_to_anns[imID], imID_to_dims)
#             assert norm_const > 0
#             for j in range(len(imID_to_anns[imID])):
#                 img_ctr = imID_to_dims[imID]
#                 x_i = imID_to_anns[imID][j]['bbox'][0] + imID_to_anns[imID][j]['bbox'][2]/2
#                 y_i = imID_to_anns[imID][j]['bbox'][1] + imID_to_anns[imID][j]['bbox'][3]/2
#                 d_i = math.sqrt((x_i - img_ctr[0])**2 + (y_i - img_ctr[1])**2)
#                 imID_to_annWeights[imID][j] = norm_const/d_i
#                 imID_to_catWeights[imID][imID_to_anns[imID][j]['category_id'] - 1] += norm_const/d_i
#     return (imID_to_catWeights, imID_to_annWeights)


# # def get_weights_per_im(imID_to_anns):

# #     imID_to_catWeights = {}
# #     imID_to_annWeights = {}
# #     for imID in imID_to_anns:
# #         # imID_to_catWeights[imID][i] will be the weight of the (i + 1)-th category as given by D['categories']
# #         # reason i + 1 is that pascal_ann.json 1-indexed the categories
# #         imID_to_catWeights[imID] = [0 for _ in range(20)]
# #         imID_to_annWeights[imID] = [0 for _ in range(len(imID_to_anns[imID]))]
# #         sum_areas = get_sum_areas(imID_to_anns[imID])
# #         assert sum_areas > 0

# #         # START new code
# #         # if bias == 'size':
# #         #     sum_areas = get_sum_areas(imID_to_anns[imID])
# #         #     assert sum_areas > 0
# #         # elif bias == 'location':
# #         #     continue
# #         # END NEW CODE
# #         for i in range(len(imID_to_anns[imID])):
# #             imID_to_catWeights[imID][imID_to_anns[imID][i]['category_id'] - 1] += imID_to_anns[imID][i]['area'] / sum_areas
# #             imID_to_annWeights[imID][i] = imID_to_anns[imID][i]['area'] / sum_areas
# #     return (imID_to_catWeights, imID_to_annWeights)


# def create_image_list(ann_dict, image_list, phase):
#     # TODO: no reason to repeat this code from format_pascal.py
#     # image_list = []
#     for cat_name in catName_to_catID:
#         with open(os.path.join(data_path + '/pascal', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', cat_name + '_' + phase + '.txt'), 'r') as f:
#             for line in f:
#                 cur_line = line.rstrip().split(' ')
#                 image_id = cur_line[0]
#                 label = cur_line[-1]
#                 image_fname = image_id + '.jpg'
#                 # if cat occurs, save it in ann_dict, which maps from image to catID that occur in the image
#                 # image_list holds all images with annotated objects in them, classified by train or val
#                 if int(label) == 1:
#                     if image_fname not in ann_dict:
#                         ann_dict[image_fname] = []
#                         image_list[phase].append(image_id)
#                     ann_dict[image_fname].append(catName_to_catID[cat_name])
#     # return image_list


# def observe_bias(label_matrix, imID_to_catWeights, phase, num_observations=1):
#     '''
#     Issue: I want to visualize biased labels. Eli's approach is to have a matrix of ocurrences.
#     So I can get the most likely observation and make a label_matrix. But I won't be able to visualize
#     from that, I need to know which specific annotation is most likely.
#     '''
#     label_matrix_biased = np.zeros_like(label_matrix)
#     (num_images, _) = np.shape(label_matrix_biased)
#     for im_id in imID_to_catWeights:
#         meta[]
#     # for row_idx in range(num_images):
#     #     im_id = meta["row_idx_to_im_id"][phase][row_idx]
#     #     cat_idx_to_weight = imID_to_catWeights[im_id]
#     #     cat_weights = np.zeros_like(label_matrix[row_idx])
#     #     for cat_idx in range(len(cat_weights)):
#     #         cat_weights[cat_idx] = cat_idx_to_weight[cat_idx]
#     #     idx_pos = int(random.choices(label_matrix[row_idx], cat_weights, k=num_observations)[0])
#     #     label_matrix_biased[row_idx][idx_pos] = 1.0
#     return label_matrix_biased


# def get_matrixIdx_to_imID(image_list_ph):
#     matrix_idx_2_im_id = {}
#     for i in range(len(image_list_ph)):
#         matrix_idx_2_im_id[i] = image_list_ph[i]
#     return matrix_idx_2_im_id


# def get_biased_annotations(imID_to_anns, imID_to_annWeights):
#     biased_annotations = {}
#     for curr_image_id in imID_to_anns:
#         curr_weights = imID_to_annWeights[curr_image_id]
#         chosen_annotation = random.choices(imID_to_anns[curr_image_id], curr_weights)
#         biased_annotations[curr_image_id] = chosen_annotation[0]
#     return biased_annotations


# # def visualize_bias(imID_to_anns, biased_annotations):
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     font_scale = 0.85
# #     color = (255, 0, 127)
# #     thickness = 2

# #     sample_size = 5
# #     indices = [random.randint(0, len(D['annotations']) - 1) for _ in range(sample_size)]
# #     sample_image_ids = [D['annotations'][idx]['image_id'] for idx in indices]
# #     im_folder_path = '/home/julioarroyo/research Eli and Julio/Biased-training-dataset-generation-for-multilabel-classification/data/pascal/VOCdevkit/VOC2012/JPEGImages/'
# #     for k in range(len(sample_image_ids)):
# #         window_name = 'Visualize biased sampling'
# #         fully_annotated = cv2.imread(im_folder_path + sample_image_ids[k] + '.jpg')
# #         for ann in imID_to_anns[sample_image_ids[k]]:
# #             top_left = (ann['bbox'][0], ann['bbox'][1])
# #             bottom_right = (ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3])
# #             fully_annotated = cv2.rectangle(fully_annotated, top_left, bottom_right, color, thickness)
# #             cv2.putText(fully_annotated, catID_to_catName[ann['category_id'] - 1], top_left, font, fontScale=font_scale, color=color, thickness=thickness, lineType=cv2.LINE_AA)        
# #         partially_annotated = cv2.imread(im_folder_path + sample_image_ids[k] + '.jpg')
# #         biased_ann = biased_annotations[sample_image_ids[k]]
# #         top_left = (biased_ann['bbox'][0], biased_ann['bbox'][1])
# #         bottom_right = (biased_ann['bbox'][0] + biased_ann['bbox'][2], biased_ann['bbox'][1] + biased_ann['bbox'][3])
# #         partially_annotated = cv2.rectangle(partially_annotated, top_left, bottom_right, color, thickness)
# #         cv2.putText(partially_annotated, catID_to_catName[biased_ann['category_id'] - 1], top_left, font, fontScale=font_scale, color=color, thickness=thickness, lineType=cv2.LINE_AA)

# #         comparison = np.concatenate((fully_annotated, partially_annotated), axis=1)
# #         cv2.imshow(window_name, comparison)
# #         cv2.waitKey()


# def test_observe_bias(imID_to_catWeights):
#     sample_size = 5
#     indices = [random.randint(0, len(D['annotations']) - 1) for _ in range(sample_size)]
#     sample_image_ids = [D['annotations'][idx]['image_id'] for idx in indices]
#     for k in range(len(sample_image_ids)):
#         curr_weights = imID_to_catWeights[sample_image_ids[k]]
#         N = 1000
#         frequencies = [0 for _ in range(20)]
#         for i in range(N):
#             idx_pos = random.choices(list(range(20)), curr_weights)[0]
#             frequencies[idx_pos] += 1
#         max_freq = float('-inf')
#         best_pos = -1
#         for j in range(len(frequencies)):
#             if frequencies[j] > max_freq:
#                 max_freq = frequencies[j]
#                 best_pos = j
#         print('RESULT {}'.format(k))
#         print('Image ID: {}'.format(sample_image_ids[k]))
#         print('Most often labeled category was {}, which was chosen with probability {}'.format(catID_to_catName[best_pos], max_freq/N))
#         print('The weights were {}'.format(curr_weights))

# def test_sampling(N, bias):
#     for phase in ['train', 'val']:
#         full_labels = np.load(os.path.join(labels_path, 'formatted_{}_labels.npy'.format(phase)))
#         misses = 0
#         for i in range(1, 1 + N):
#             single_labels = np.load(os.path.join(labels_path, f'coco_formatted_{phase}_{bias}_{i}_labels_obs.npy'))
#             print('SINGLE SHAPE {} FULL {}'.format(single_labels.shape, full_labels.shape))
#             assert full_labels.shape == single_labels.shape, f'{full_labels.shape} {single_labels.shape} phase {phase}'
#             rows, cols = single_labels.shape
#             print(f'single_labels.shape {single_labels.shape}')
#             for r in range(rows):
#                 for c in range(cols):
#                     if single_labels[r][c] == 1:
#                         if not full_labels[r][c] == 1:
#                             misses += 1
#         print('misses {} for {}'.format(misses, phase))
#     return



# if __name__ == '__main__':
#     mode = 'test'  # 'test' or 'run'
#     bias = 'size'

#     imID_to_anns = get_anns_by_imID()
#     # (cat_list, cat_id_to_index) = parse_categories()
#     (imID_to_catWeights, _) = get_weights_per_im(imID_to_anns, bias)

#     N = 1

#     if mode == 'test':
#         test_sampling(N, bias)
#     else:
#         ann_dict = {}
#         image_list = {'train': [], 'val': []}
        
        
#         for i in range(1, N + 1):
#             for phase in ['train', 'val']:
#                 base_path = data_path + 'coco/'
#                 label_matrix = np.load(os.path.join('/home/julioarroyo/research_Eli_and_Julio/single-positive-multi-label-julio/data/coco', 'formatted_{}_labels.npy'.format(phase)))
#                 assert np.max(label_matrix) == 1
#                 assert np.min(label_matrix) == 0

#                 # image_list_ph = create_image_list(ann_dict, phase)
#                 # create_image_list(ann_dict, image_list, phase)
#                 # assert len(image_list[phase]) > 0
#                 # image_list[phase].sort()
#                 # matrix_idx_2_im_id = get_matrixIdx_to_imID(image_list)

#                 # convert label matrix to -1 / +1 format:
#                 label_matrix[label_matrix == 0] = -1

#                 random.seed(SEED)
#                 biased_matrix = observe_bias(label_matrix, imID_to_catWeights, phase)
#                 assert label_matrix.shape == biased_matrix.shape, f'{label_matrix} {biased_matrix}'
#                 np.save(os.path.join('/home/julioarroyo/research_Eli_and_Julio/single-positive-multi-label-julio/data/coco', 'coco_formatted_{}_{}_{}_labels_obs.npy'.format(phase, bias, i)), biased_matrix)
        
#         # biased_dataset = get_biased_annotations(imID_to_anns, imID_to_annWeights)
#         # visualize_bias(imID_to_anns, biased_dataset)



