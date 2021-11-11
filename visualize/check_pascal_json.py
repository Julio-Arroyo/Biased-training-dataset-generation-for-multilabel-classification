import json
import cv2
import random

if __name__ == '__main__':
    pascal_json_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Research with Eli Cole/Biased Dataset Generation/data/pascal/pascal_cocoformat/pascal_ann.json'
    f = open(pascal_json_path)
    data = json.load(f)
    indices = [random.randint(0, len(data['annotations']) - 1) for _ in range(5)]
    sample_image_ids = [data['annotations'][idx]['image_id'] for idx in indices]
    im_folder_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Research with Eli Cole/Biased Dataset Generation/data/pascal/VOCdevkit/VOC2012/JPEGImages/'
    for k in range(len(indices)):
        im = cv2.imread(im_folder_path + sample_image_ids[k] + '.jpg')
        window_name = 'test .json annotations'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.85
        top_left = (data['annotations'][indices[k]]['bbox'][0], data['annotations'][indices[k]]['bbox'][1])
        bottom_right = (data['annotations'][indices[k]]['bbox'][0] + data['annotations'][indices[k]]['bbox'][2], data['annotations'][indices[k]]['bbox'][1] + data['annotations'][indices[k]]['bbox'][3])
        color = (255, 0, 127)
        thickness = 2
        im = cv2.rectangle(im, top_left, bottom_right, color, thickness)
        cat_id = data['annotations'][indices[k]]['category_id']
        name = None
        for i in range(len(data['categories'])):
            if data['categories'][i]['id'] == cat_id:
                name = data['categories'][i]['name']
                break
        cv2.putText(im, name, top_left, font, fontScale=font_scale, color=color, thickness= thickness, lineType=cv2.LINE_AA)
        cv2.imshow(window_name, im)
        cv2.waitKey()
