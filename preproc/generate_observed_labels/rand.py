import json

data_path = '/media/julioarroyo/aspen/data'
coco_json_path = '/coco/annotations/instances_train2014.json'
f = open(data_path + coco_json_path)
D = json.load(f)
print('annotations' in D)