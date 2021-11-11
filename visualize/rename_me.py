import cv2
import numpy as np

# Goal: pick any image at random, visualize all its annotations
# need: cat_id2name dictionary
# need: dict from image to all its annotations
# try: can you make an nxm grid of images

def make_image_grid(im_path):
    im = cv2.imread(im_path)
    row1 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row2 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row3 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row4 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row5 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row6 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row7 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row8 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row9 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    row10 = np.concatenate((im, im, im, im, im, im, im, im, im, im), axis=1)
    cols = np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8, row9, row10), axis=0)
    cv2.imshow('TEST IMAGE ARRAY', cols)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    make_image_grid('/Users/jarroyo/OneDrive - California Institute of Technology/Research with Eli Cole/Biased Dataset Generation/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_003118.jpg')
