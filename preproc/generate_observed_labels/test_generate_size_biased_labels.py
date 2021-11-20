import generate_size_biased_labels as GSBL
import numpy as np
import random
import json
import os


data_path = 'PATH TO DATA'
pascal_json_path = 'pascal/pascal_cocoformat/pascal_ann.json'
f = open(data_path + pascal_json_path)
D = json.load(f)
SEED = 10


def get_im_id_2_matrix_idx(matrix_idx_2_im_id):
    im_id_2_matrix_idx = {matrix_idx_2_im_id[k]: k for k in matrix_idx_2_im_id}
    return im_id_2_matrix_idx


def test_generated_matrix(imID_to_catWeights):
    im_id_2_matrix_idx = get_im_id_2_matrix_idx(GSBL.matrix_idx_2_im_id)
    sample_size = 5
    indices = [random.randint(0, len(D['annotations']) - 1) for _ in range(sample_size)]
    sample_image_ids = [D['annotations'][idx]['image_id'] for idx in indices]
    for k in range(len(sample_image_ids)):
        curr_weights = imID_to_catWeights[sample_image_ids[k]]
        matrix_curr_weights = im
            
    base_path = '/home/julioarroyo/research Eli and Julio/Biased-training-dataset-generation-for-multilabel-classification/data/pascal/'
    i = 1
    phase = 'train'
    tested_matrix = np.load(os.path.join(base_path, 'formatted_{}{}_size_bias_labels.npy'.format(phase, i)))
    (num_images, num_classes) = np.shape(tested_matrix)
    for row_idx in range(num_images):
        curr_im_id = GSBL.matrix_idx_2_im_id[row_idx]
        cat_weights = GSBL.imID_to_catWeights[curr_im_id]
        # TODO: this in theory is general for any num_observation, problem is random.choices can repeat
        idx_pos = random.choices(list(range(20)), cat_weights, k=1)
        label_matrix_biased[row_idx, idx_pos] = 1.0
    return label_matrix_biased


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