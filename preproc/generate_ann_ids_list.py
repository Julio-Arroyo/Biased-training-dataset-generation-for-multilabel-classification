from os import close, listdir, write
from os.path import isfile, join, splitext

if __name__ == '__main__':
    ann_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Research with Eli Cole/Biased Dataset Generation/data/pascal/VOCdevkit/VOC2012/Annotations'
    all_ids = [splitext(fname)[0] for fname in listdir(ann_path)]
    txt_file_name = 'pascal_ids.txt'
    dst_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Research with Eli Cole/Biased Dataset Generation/data/pascal/pascal_cocoformat'
    txt_file = open(join(dst_path, txt_file_name), 'w')
    for curr_id in all_ids:
        txt_file.write(curr_id + '\n')
    txt_file.close()
