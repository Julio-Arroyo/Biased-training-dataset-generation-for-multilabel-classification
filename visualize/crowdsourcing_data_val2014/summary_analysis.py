import matplotlib.pyplot as plt
import json
import statistics


if __name__ == '__main__':
    cat_list = open('categories.json')
    cat_list = json.load(cat_list)
    # print(cat_list)
    cat_id_2_cat_name = {}  # map: cat_id --> cat_name
    cat_name_2_cat_id = {}
    for entry in cat_list['categories']:
        cat_id_2_cat_name[entry['id']] = entry['name']
        cat_name_2_cat_id[entry['name']] = entry['id']
    
    #  How much data
    f_annotate = open('crowdsourcing_annotate_category_val2014.json')
    annotations = json.load(f_annotate)
    print('Number of entries in annotate: {}'.format(len(annotations)))
    cat_2_ann_num = {}
    im_id_2_num_anns = {}
    for annotation in annotations:
        cat_name = cat_id_2_cat_name[annotation['cat_id']]
        if cat_name not in cat_2_ann_num:
            cat_2_ann_num[cat_name] = 0
        cat_2_ann_num[cat_name] += 1

        if annotation['im_id'] not in im_id_2_num_anns:
            im_id_2_num_anns[annotation['im_id']] = 0
        im_id_2_num_anns[annotation['im_id']] += 1
    cat_name_list = list(cat_2_ann_num.keys())
    vals = list(cat_2_ann_num.values())
    print("AVG: {}".format(sum(list(im_id_2_num_anns.values()))/len(im_id_2_num_anns)))
    print("MEDIAN: {}".format(statistics.median(list(im_id_2_num_anns.values()))))
    # print(vals)
    # cat_id_list = [cat_name_2_cat_id[cat_name] for cat_name in cat_name_list]
    plt.bar(cat_name_list, vals)
    plt.xticks(rotation = 90)
    fig = plt.gcf()
    fig.set_size_inches(24, 10.5)
    fig.savefig('distribution.png', dpi=100)
    # plt.gcf().subplots_adjust(left=0, right=10)
    # plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        
