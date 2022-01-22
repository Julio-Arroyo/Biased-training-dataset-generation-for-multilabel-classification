import matplotlib.pyplot as plt
import numpy as np

loss_2_bias = {'AN' : {'size': [[], []], 'location': [[], []]}, 
                'AN LS' : {'size': [[], []], 'location': [[], []]}, 
                'ROLE' : {'size': [[], []], 'location': [[], []]}}

f = open('experimental_data.txt', 'r')
data = f.readlines()
for datapoint in data:
    vals = datapoint.split(',')
    if 'size' in vals[0].lower():
        bias = 'size'
    elif 'location' in vals[0].lower():
        bias = 'location'
    else:
        assert False
    
    loss_2_bias[vals[3]][bias][0].append(float(vals[1])) # append test value
    loss_2_bias[vals[3]][bias][1].append(float(vals[2])) # append validation value

for loss in loss_2_bias:
    for i in range(2):
        if i == 0:
            phase = 'test'
        else: # i == 1
            phase = 'val'
        map_vals = [loss_2_bias[loss]['size'][i], loss_2_bias[loss]['location'][i]]
        
        
        fig, ax = plt.subplots()
        fig.canvas.draw()
        ax.set_xticklabels(['Size', 'Location'])
        title = '{}_{}'.format(loss, phase)
        fig.suptitle(title)
        plt.xlabel('Bias Type')
        plt.ylabel('MAP')
        plt.boxplot(map_vals)
        plt.savefig(title + '.jpg')