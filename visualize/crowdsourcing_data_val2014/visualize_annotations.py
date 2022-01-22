import pandas as pd
import json
import requests
from matplotlib import image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import matplotlib.image as mpimg

cat_list = open('categories.json')
cat_list = json.load(cat_list)
# print(cat_list)
cat_id_2_cat_name = {}  # map: cat_id --> cat_name
cat_name_2_cat_id = {}
for entry in cat_list['categories']:
    cat_id_2_cat_name[entry['id']] = entry['name']
    cat_name_2_cat_id[entry['name']] = entry['id']

def get_images():
    return [569652, 381615]

img_list = get_images()

f_annotate = open('crowdsourcing_annotate_category_val2014.json')
annotations = json.load(f_annotate)

img_2_cat_2_xy = {}

num_cats = 91
for img_id in img_list:
    cat_2_xy = {}
    for cat_id in range(1, num_cats + 1):
        cat_2_xy[cat_id] = [[], []]
    img_2_cat_2_xy[img_id] = cat_2_xy

for annotation in annotations:
    if annotation["im_id"] in img_2_cat_2_xy:
        # if annotation["cat_id"] not in img_2_cat_2_xy[annotation["im_id"]]:
        #     img_2_cat_2_xy[annotation["im_id"]][annotation["cat_id"]] = 
        im_id = annotation["im_id"]
        cat_id = annotation["cat_id"]
        
        img_2_cat_2_xy[im_id][cat_id][0].append(annotation["x"])
        img_2_cat_2_xy[im_id][cat_id][1].append(annotation["y"])
# print(img_2_cat_2_xy)

coco = COCO("/home/julioarroyo/Downloads/annotations/instances_val2014.json")
imgs_data = coco.loadImgs(img_list)

# GET IMAGES FROM INTERNET
# for im in imgs_data:
#     img_data = requests.get(im['coco_url']).content
#     with open('/home/julioarroyo/research_Eli_and_Julio/single-positive-multi-label-julio/visualize/images/sample_COCO/' + im['file_name'], 'wb') as handler:
#         handler.write(img_data)

for i in range(len(img_list)):
    plt.figure(figsize=(15, 6), dpi=80)
    img_arr = mpimg.imread("/home/julioarroyo/research_Eli_and_Julio/single-positive-multi-label-julio/visualize/images/sample_COCO/COCO_val2014_000000{}.jpg".format(img_list[i]))
    for cat in img_2_cat_2_xy[img_list[i]]:
        X = img_2_cat_2_xy[img_list[i]][cat][0]
        X = [X[k]*img_arr.shape[1] for k in range(len(X))]
        Y = img_2_cat_2_xy[img_list[i]][cat][1]
        Y = [Y[r]*img_arr.shape[0] for r in range(len(Y))]
        if len(X) > 0 and len(Y) > 0:
            plt.scatter(X, Y, label=cat_id_2_cat_name[cat])
            plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.imshow(img_arr)
    # plt.show()
    plt.savefig("{}_ann.jpg".format(img_list[i]))


# image_list = get_images()

# f_annotate = open('crowdsourcing_annotate_category_val2014.json')
# annotations = json.load(f_annotate)
# x = []
# y = []
# cat_ids = []
# image_ids = []

# for annotation in annotations:
#     x.append(annotation['x'])
#     y.append(annotation['y'])
#     cat_ids.append(annotation['cat_id'])
#     image_ids.append(annotation['image_id'])

# df = pd.DataFrame({'x':x, 'y':y, 'cat_id':cat_ids, 'image_ids':image_ids})

# image_groups = df.groupby('image_id')

# for name, group in image_groups:
#     group.


# for every image
    # plot every category

# for image_name in image_list:
#     img = image.imread(image_name)

#     # to draw a point on co-ordinate (200,300)
#     plt.plot(img.shape[0], img.shape[1], marker='x', color="white")
#     plt.imshow(img)
#     plt.show()