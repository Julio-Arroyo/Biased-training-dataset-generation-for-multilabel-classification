'''
GOAL: show two images side by side,  fully annotated and biased-sampled (by size) annotated, 
print statistics how many times that happened.
'''


import os
# import cv2
import json
import random
import numpy as np
from format_pascal import catID_to_catName
from format_pascal import catName_to_catID

data_path = '/home/julioarroyo/research Eli and Julio/single-positive-multi-label-julio/data/'
pascal_json_path = 'pascal/pascal_cocoformat/pascal_ann.json'
f = open(data_path + pascal_json_path)
D = json.load(f)
SEED = 10


def get_anns_by_imID(im_to_cat):
    imID_to_anns = {}
    for i in range(len(D['annotations'])):
        if D['annotations'][i]['image_id'] in im_to_cat and ((D['annotations'][i]['category_id'] - 1) in im_to_cat[D['annotations'][i]['image_id']]):
            if not D['annotations'][i]['image_id'] in imID_to_anns:
                imID_to_anns[D['annotations'][i]['image_id']] = []
            imID_to_anns[D['annotations'][i]['image_id']].append(D['annotations'][i])
    f.close()
    return imID_to_anns


def get_sum_areas(annotations):
    sum_area = 0
    assert len(annotations) > 0
    for ann in annotations:
        sum_area += ann['area']
    return sum_area


def get_weights_per_im(imID_to_anns):
    '''
    Return a map from image id to weights.

    Test:
        - there is a non-zero weights if and only if there is an annotation
    '''
    assert(len(imID_to_anns) > 0)
    imID_to_catWeights = {}
    imID_to_annWeights = {}
    for imID in imID_to_anns:
        # imID_to_catWeights[imID][i] will be the weight of the (i + 1)-th category as given by D['categories']
        # reason i + 1 is that pascal_ann.json 1-indexed the categories
        imID_to_catWeights[imID] = [0 for _ in range(20)]
        imID_to_annWeights[imID] = [0 for _ in range(len(imID_to_anns[imID]))]
        sum_areas = get_sum_areas(imID_to_anns[imID])
        assert sum_areas > 0

        for i in range(len(imID_to_anns[imID])):
            imID_to_catWeights[imID][imID_to_anns[imID][i]['category_id'] - 1] += imID_to_anns[imID][i]['area'] / sum_areas
            imID_to_annWeights[imID][i] = imID_to_anns[imID][i]['area'] / sum_areas
    
    # Test weight != 0 iff there is annotation
    for im_id in imID_to_catWeights:
        curr_cat_weights = imID_to_catWeights[im_id]
        for i in range(len(curr_cat_weights)):
            if curr_cat_weights[i] != 0:
                found_ann = False
                for ann in imID_to_anns[im_id]:
                    if ann['category_id'] - 1 == i:
                        found_ann = True
                        break
                if not found_ann:
                    print('Non-zero weight index i={}'.format(i))
                    print('image id {}'.format(im_id))
                    assert False
    return (imID_to_catWeights, imID_to_annWeights)


def create_image_list():
    # TODO: no reason to repeat this code from format_pascal.py
    # image_list = []
    # for cat_name in catName_to_catID:
    #     with open(os.path.join(data_path + '/pascal', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', cat_name + '_' + phase + '.txt'), 'r') as f:
    #         for line in f:
    #             cur_line = line.rstrip().split(' ')
    #             image_id = cur_line[0]
    #             label = cur_line[-1]
    #             image_fname = image_id + '.jpg'
    #             # if cat occurs, save it in ann_dict, which maps from image to catID that occur in the image
    #             # image_list holds all images with annotated objects in them, classified by train or val
    #             if int(label) == 1:
    #                 if image_fname not in ann_dict:
    #                     ann_dict[image_fname] = []
    #                     image_list.append(image_id)
    #                 ann_dict[image_fname].append(catName_to_catID[cat_name])
    ann_dict = {}
    image_list = {'train': [], 'val': []}

    for phase in ['train', 'val']:
        for cat in catName_to_catID:
            # this file tells you in what images does cat occurr
            with open(os.path.join(data_path + '/pascal', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', cat + '_' + phase + '.txt'), 'r') as f:
                for line in f:
                    cur_line = line.rstrip().split(' ')
                    image_id = cur_line[0]
                    label = cur_line[-1]
                    # if cat occurs, save it in ann_dict, which maps from image to catID that occur in the image
                    # image_list holds all images with annotated objects in them, classified by train or val
                    if int(label) == 1:
                        if image_id not in ann_dict:
                            ann_dict[image_id] = []
                            image_list[phase].append(image_id)
                        ann_dict[image_id].append(catName_to_catID[cat])
    image_list['train'].sort()
    image_list['val'].sort()
    return (ann_dict, image_list)


def observe_bias(label_matrix, imID_to_catWeights, image_list_ph, imID_to_anns, num_observations=1):
    '''
    Issue: I want to visualize biased labels. Eli's approach is to have a matrix of ocurrences.
    So I can get the most likely observation and make a label_matrix. But I won't be able to visualize
    from that, I need to know which specific annotation is most likely.
    '''
    label_matrix_biased = np.zeros_like(label_matrix)
    (num_images, num_classes) = np.shape(label_matrix_biased)
    disagreements = 0
    for row_idx in range(num_images):
        curr_im_id = image_list_ph[row_idx]
        if not curr_im_id in imID_to_catWeights:
            print('len of cat weights dict: {}'.format(len(imID_to_catWeights)))
            print('FAIL AT row_idx = {}'.format(row_idx))
        cat_weights = imID_to_catWeights[curr_im_id]

        # Test annotations obtained from annotation file agree with label_matrix
        for i in range(len(cat_weights)):
            if cat_weights[i] != 0:
                if label_matrix[row_idx, i] != 1:        
                    label_indices = [k for k in range(len(label_matrix[row_idx, :])) if label_matrix[row_idx, k] == 1]
                    ann_file_label_indices = set()
                    for l in range(len(imID_to_anns[curr_im_id])):
                        ann_file_label_indices.add(imID_to_anns[curr_im_id][l]['category_id'] - 1)
                    ann_file_label_indices = list(ann_file_label_indices)
                    ann_file_label_indices.sort()
                    if ann_file_label_indices != label_indices:
                        disagreements += 1
                        print('{}. image id {}. Annotation file {}. Eli matrix {}'.format(disagreements, curr_im_id, ann_file_label_indices, label_indices))

        # # Test cat_weights make sense:
        # for i in range(len(cat_weights)):
        #     if cat_weights[i] != 0:
        #         if label_matrix[row_idx, i] != 1:
        #             print('****')
        #             print('There is a weight for a non-observed category in image {} and row_idx {}'.format(curr_im_id, row_idx))
        #             present_indices = [j for j in range(len(cat_weights)) if cat_weights[j] != 0]
        #             label_indices = [k for k in range(len(label_matrix[row_idx, :])) if label_matrix[row_idx, k] == 1]
        #             print('label vector indices {}'.format(label_indices))
        #             print('cat_weights indices {}'.format(present_indices))
        #             print('problematic weight is {} at i={}'.format(cat_weights[i], i))
        #             print('****')
        # TODO: this in theory is general for any num_observation, problem is random.choices can repeat
        idx_pos = random.choices(list(range(20)), cat_weights, k=num_observations)[0]
        # assert label_matrix[row_idx, idx_pos] == 1, 'ERROR at {}, {}'.format(row_idx, idx_pos)
        label_matrix_biased[row_idx, idx_pos] = 1.0
    
    # for i in range(num_images):
    #     for j in range(num_classes):
    #         if label_matrix_biased[i, j] == 1:
    #             assert label_matrix[i, j] == 1, 'ERROR at {}, {}'.format(i, j)
    # for row_idx in range(num_images):
    #     curr_im_id = matrix_idx_2_im_id[row_idx]
    #     cat_weights = imID_to_catWeights[curr_im_id]
    #     # TODO: this in theory is general for any num_observation, problem is random.choices can repeat
    #     idx_pos = random.choices(list(range(20)), cat_weights, k=num_observations)[0]
    #     label_matrix_biased[row_idx, idx_pos] = 1.0
    return label_matrix_biased


def get_matrixIdx_to_imID(image_list):
    matrix_idx_2_im_id = {'train': {}, 'val': {}}
    for phase in ['train', 'val']:
        for i in range(len(image_list[phase])):
            matrix_idx_2_im_id[phase][i] = image_list[phase][i]
    return matrix_idx_2_im_id


def get_biased_annotations(imID_to_anns, imID_to_annWeights):
    biased_annotations = {}
    for curr_image_id in imID_to_anns:
        curr_weights = imID_to_annWeights[curr_image_id]
        chosen_annotation = random.choices(imID_to_anns[curr_image_id], curr_weights)
        biased_annotations[curr_image_id] = chosen_annotation[0]
    return biased_annotations


# def visualize_bias(imID_to_anns, biased_annotations):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.85
#     color = (255, 0, 127)
#     thickness = 2

#     sample_size = 5
#     indices = [random.randint(0, len(D['annotations']) - 1) for _ in range(sample_size)]
#     sample_image_ids = [D['annotations'][idx]['image_id'] for idx in indices]
#     im_folder_path = '/home/julioarroyo/research Eli and Julio/Biased-training-dataset-generation-for-multilabel-classification/data/pascal/VOCdevkit/VOC2012/JPEGImages/'
#     for k in range(len(sample_image_ids)):
#         window_name = 'Visualize biased sampling'
#         fully_annotated = cv2.imread(im_folder_path + sample_image_ids[k] + '.jpg')
#         for ann in imID_to_anns[sample_image_ids[k]]:
#             top_left = (ann['bbox'][0], ann['bbox'][1])
#             bottom_right = (ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3])
#             fully_annotated = cv2.rectangle(fully_annotated, top_left, bottom_right, color, thickness)
#             cv2.putText(fully_annotated, catID_to_catName[ann['category_id'] - 1], top_left, font, fontScale=font_scale, color=color, thickness=thickness, lineType=cv2.LINE_AA)        
#         partially_annotated = cv2.imread(im_folder_path + sample_image_ids[k] + '.jpg')
#         biased_ann = biased_annotations[sample_image_ids[k]]
#         top_left = (biased_ann['bbox'][0], biased_ann['bbox'][1])
#         bottom_right = (biased_ann['bbox'][0] + biased_ann['bbox'][2], biased_ann['bbox'][1] + biased_ann['bbox'][3])
#         partially_annotated = cv2.rectangle(partially_annotated, top_left, bottom_right, color, thickness)
#         cv2.putText(partially_annotated, catID_to_catName[biased_ann['category_id'] - 1], top_left, font, fontScale=font_scale, color=color, thickness=thickness, lineType=cv2.LINE_AA)

#         comparison = np.concatenate((fully_annotated, partially_annotated), axis=1)
#         cv2.imshow(window_name, comparison)
#         cv2.waitKey()


def test_observe_bias(imID_to_catWeights):
    sample_size = 5
    indices = [random.randint(0, len(D['annotations']) - 1) for _ in range(sample_size)]
    sample_image_ids = [D['annotations'][idx]['image_id'] for idx in indices]
    for k in range(len(sample_image_ids)):
        curr_weights = imID_to_catWeights[sample_image_ids[k]]
        N = 1000
        frequencies = [0 for _ in range(20)]
        for i in range(N):
            idx_pos = random.choices(list(range(20)), curr_weights)[0]
            frequencies[idx_pos] += 1
        max_freq = float('-inf')
        best_pos = -1
        for j in range(len(frequencies)):
            if frequencies[j] > max_freq:
                max_freq = frequencies[j]
                best_pos = j
        print('RESULT {}'.format(k))
        print('Image ID: {}'.format(sample_image_ids[k]))
        print('Most often labeled category was {}, which was chosen with probability {}'.format(catID_to_catName[best_pos], max_freq/N))
        print('The weights were {}'.format(curr_weights))




if __name__ == '__main__':
    mode = 'run'  # 'test' or 'run'

    (im_to_cat, image_list) = create_image_list()
    imID_to_anns = get_anns_by_imID(im_to_cat)


    # TODO: add a parameter below for kind of bias
    (imID_to_catWeights, imID_to_annWeights) = get_weights_per_im(imID_to_anns)

    if mode == 'test':
        test_observe_bias(imID_to_catWeights)
    else:
        N = 5
        for i in range(1, N + 1):
            for phase in ['train', 'val']:
                base_path = data_path + 'pascal/'
                label_matrix = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
                
                assert np.max(label_matrix) == 1
                assert np.min(label_matrix) == 0

                # convert label matrix to -1 / +1 format:
                label_matrix[label_matrix == 0] = -1

                random.seed(SEED)
                biased_matrix = observe_bias(label_matrix, imID_to_catWeights, image_list[phase], imID_to_anns)

                np.save(os.path.join(base_path, 'formatted_{}_labels_obs_{}.npy'.format(phase, i)), biased_matrix)
        
        biased_dataset = get_biased_annotations(imID_to_anns, imID_to_annWeights)
        # visualize_bias(imID_to_anns, biased_dataset)


# 5 REALIZATIONS OF THE SAME DATASET
# MEAN average precision
# multilabel classification metrics: how do people evaluate, what are the pros and cons
# get code running with pascal, standard configuration and match on paper

# multilabel classification metrics:
    # precision
    # recall
    # F1: balance between recall and precision
    # 0/1 loss: 1 if not equal, 0 otherwise.
    # Hamming Score/ Accuracy: is the fraction of correct predictions compared to the total labels.
    # Hamming Loss: 

    #Scikit learn:
        # Coverage error:
            # average number of labels that have to be included in the final prediction such that all true labels are predicted.
            # useful if you want to know how many top-scored-labels you have to predict in average without missing any true one. 
            # The best value of this metrics is thus the average number of true labels.
        # LRAP (Label ranking average precision): basically measures how many true labels we ranked at what ranking
        # Ranking loss:
    

    # Multilabel classification an overview:
        # hamming loss
        # accuracy
        # precision
        # recall
        # MAP: area under precision-recall curve
