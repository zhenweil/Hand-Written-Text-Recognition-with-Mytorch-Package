import numpy as np
import skimage
import skimage.io
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.filters import gaussian
from skimage.morphology import opening, binary_erosion, binary_dilation, closing
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu, threshold_mean, threshold_minimum
from skimage.transform import resize, downscale_local_mean, rescale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import os
import string
# takes a color image
# returns a list of bounding boxes and black_and_white image
def sort_bbox_row(val):
    return val[0]
def sort_bbox_col(val):
    return val[1]
def plot(img):
    plt.plot()
    plt.imshow(img, cmap = "gray")
    plt.show()
def findLetters(image):
    bboxes = []
    bw = None
    gray = rgb2gray(image)
    blurred = gaussian(gray, 3)
    thresh_value = threshold_otsu(blurred)
    binary = blurred > thresh_value
    binary = closing(binary)
    plot(binary)
    label_img, num = label(binary, neighbors = 8, background = 1, return_num = True)
    image_overlay = label2rgb(label_img, image = image, bg_label=0)

    for region in regionprops(label_img):
        if region.area >= 200:
            minr,minc,maxr,maxc = region.bbox
            rect = mpatches.Rectangle((minc,minr),maxc-minc,maxr-minr,fill=False,
                                       edgecolor = 'red', linewidth=1)
            one_box = [minr,minc,maxr,maxc]
            bboxes.append(one_box)

    bboxes.sort(key = sort_bbox_row)
    minr_most = bboxes[0][0]
    dist_thresh = 50
    same_row = []
    row_sorted = []
    for bbox in bboxes:
        this_minr = bbox[0]
        if np.abs(this_minr - minr_most) < dist_thresh:
            same_row.append(bbox) 
        else:
            row_sorted.append(deepcopy(same_row))
            same_row.clear()
            same_row.append(bbox)
        minr_most = bbox[0]
        if bbox == bboxes[-1]:
            row_sorted.append(deepcopy(same_row))
        pass
       
    sorted_bboxes = []
    for each_row_sorted in row_sorted:
        each_row_sorted.sort(key = sort_bbox_col)
        sorted_bboxes.append(each_row_sorted)
    
    return sorted_bboxes, binary

def get_letter_from_img(path):
    im1 = skimage.img_as_float(skimage.io.imread(path))
    bboxes, bw = findLetters(im1)
    plt.figure()
    plt.imshow(bw, cmap = 'gray')
    for row in bboxes:
        for bbox in row:
            minr, minc, maxr, maxc = bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    plt.show()
    img_data= []
    spaces = []
    for row in bboxes:
        row_data = []
        space = []
        prev_c = 0
        for bbox in row:
            minr = bbox[0]
            minc = bbox[1]
            maxr = bbox[2]
            maxc = bbox[3]
            if(prev_c != 0 and minc - prev_c > 120):
                space.append(' ')
            else:
                space.append('')
            prev_c = maxc
            patch = bw[minr:maxr,minc:maxc]
            downscale_r = (maxr-minr)//60
            downscale_c = (maxc-minc)//60
            if(downscale_r < 1):
                downscale_r = 1
            if(downscale_c < 1):
                downscale_c = 1
            patch = downscale_local_mean(patch,(downscale_r, downscale_c))
            patch = resize(patch, (20,20))
            patch = np.pad(patch,((6,6),(6,6)),mode='constant', constant_values = (1,1))
            plot(patch)
            data = patch.T
            data = data.reshape(-1,)
            row_data.append(data)
            pass
        img_data.append(row_data)
        spaces.append(space)
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    return spaces, img_data