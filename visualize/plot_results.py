import matplotlib.pyplot as plt
import numpy as np

loss_2_bias = {'AN' : {'size': [[], []], 'location': [[], []], 'uniform': [[], []]}, 
                'AN LS' : {'size': [[], []], 'location': [[], []], 'uniform': [[], []]}, 
                'ROLE' : {'size': [[], []], 'location': [[], []], 'uniform': [[], []]}}

f = open('experimental_data2.txt', 'r')
data = f.readlines()
for datapoint in data:
    datapoint = datapoint.strip()
    vals = datapoint.split(',')
    if '1' in vals[0].lower():
        bias = 'size'
    elif '2' in vals[0].lower():
        bias = 'location'
    elif '3' in vals[0].lower():
        bias = 'uniform'
    else:
        assert False
    
    loss_2_bias[vals[3]][bias][0].append(float(vals[1])) # append test value
    loss_2_bias[vals[3]][bias][1].append(float(vals[2])) # append validation value


for i in range(2):
    if i == 0:
        phase = 'test'
    else: # i == 1
        phase = 'val'
    fig, ax = plt.subplots()
    fig.canvas.draw()
    title = '{}'.format(phase)
    fig.suptitle(title)
    plt.xlabel('Bias Type')
    plt.ylabel('MAP')
    for loss in loss_2_bias:
        map_vals = [loss_2_bias[loss]['size'][i], loss_2_bias[loss]['location'][i], loss_2_bias[loss]['uniform'][i]]
        
        loss_data = np.concatenate((map_vals[0], map_vals[1], map_vals[2]))
        x_data = np.concatenate(([1]*5, [2]*5, [3]*5))
        plt.scatter(x_data, loss_data, label=loss)
        # plt.legend(['Size Bias', 'Location Bias', 'Uniform Sampling'])
    plt.xticks([1,2,3], ['Size', 'Location', 'Uniform'])
    plt.yticks(np.arange(82, 86.8, 0.25))
    plt.legend()
    plt.savefig("all_together_{}.png".format(phase))