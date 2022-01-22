import json
import matplotlib.pyplot as plt

data_path = '/media/julioarroyo/aspen/data'
coco_path = '/coco/annotations/'

def get_dicts(annotations):
    cat_to_imgs = {}
    im_id_to_anns = {}
    for annotation in annotations:
        im_id = annotation['im_id']
        cat_id = annotation['cat_id']
        if cat_id not in cat_to_imgs:
            cat_to_imgs[cat_id] = []
        cat_to_imgs[cat_id].append(im_id)
        im_id_to_anns[im_id] = annotation
    return (cat_to_imgs, im_id_to_anns)


def get_imID_to_bbox():
    imID_to_bbox = {}
    f_val = open(data_path + coco_path + 'instances_val2014.json')
    for phase in ['val']:
        D = json.load(f_val)
        for i in range(len(D['annotations'])):
            imID_to_bbox[D['annotations'][i]['image_id']] = D['annotations'][i]['bbox']
    f_val.close()
    return imID_to_bbox


def get_imID_to_dims():
    imID_to_dims = {}
    for phase in ['val']:
        f = open(data_path + coco_path + 'instances_{}2014.json'.format(phase))
        D = json.load(f)
        for i in range(len(D['images'])):
            imID_to_dims[D['images'][i]['id']] = [D['images'][i]['width'], D['images'][i]['height']]
        f.close()
    return imID_to_dims


def point_in_bbox(x, y, im_id, imID_to_bbox):
    bbox = imID_to_bbox[im_id]
    if x >= bbox[0] and x <= bbox[2] and y >= bbox[1] and y <= bbox[2]:
        return True
    return False


def get_cat_id_to_cat_name():
    cat_list = open('categories.json')
    cat_list = json.load(cat_list)
    # print(cat_list)
    cat_id_2_cat_name = {}  # map: cat_id --> cat_name
    cat_name_2_cat_id = {}
    for entry in cat_list['categories']:
        cat_id_2_cat_name[entry['id']] = entry['name']
        cat_name_2_cat_id[entry['name']] = entry['id']
    return cat_id_2_cat_name


if __name__ == '__main__':
    mode = 'run'  # 'run' or 'check'

    if mode == 'run':
        f = open('crowdsourcing_instance_spotting_val2014.json')
        annotations = json.load(f)

        (cat_to_imgs, im_id_to_anns) = get_dicts(annotations)
        im_id_to_bbox = get_imID_to_bbox()
        im_id_to_dims = get_imID_to_dims()
        cat_id_to_cat_name = get_cat_id_to_cat_name()
        cat_to_prob = {}
        iterations = 0
        for cat in cat_to_imgs:
            good_clicks = 0
            total_clicks = 0
            for im_id in cat_to_imgs[cat]:
                ann = im_id_to_anns[im_id]
                x = im_id_to_dims[im_id][0] * ann['x']
                y = im_id_to_dims[im_id][1] * ann['y']
                cat_id = ann['cat_id']
                try:
                    if cat_id == cat and point_in_bbox(x, y, im_id, im_id_to_bbox):
                        good_clicks += 1
                    total_clicks += 1
                except KeyError:
                    iterations += 1
                    continue
            cat_to_prob[cat] = good_clicks/total_clicks
        print(f'DIDNT FIND {iterations} out of {len(annotations)}')
        cat_name_list = [cat_id_to_cat_name[cat] for cat in cat_to_prob]
        vals = list(cat_to_prob.values())
        plt.bar(cat_name_list, vals)
        plt.xticks(rotation = 90)
        fig = plt.gcf()
        fig.set_size_inches(24, 10.5)
        fig.savefig('prob_model.png', dpi=100)
    elif mode == 'check':
        visual_check()
