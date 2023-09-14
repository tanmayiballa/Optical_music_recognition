#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageDraw, ImageFont
import sys
#import matplotlib.pyplot as plt
import copy

## Reference: http://www.people.vcu.edu/~bhammel/theory/resources/lessons/pitch3.htm#:~:text=The%20musical%20alphabet%20includes%20only,centered%20on%20the%20second%20line.
## Reference: https://github.com/ojaashampiholi/Optical_Music_Recognition/blob/main/Hough_Transform.py
## Dictionary for detecting the symbols of the filled notes.
## Dictionary is created based on the following references.
## The dictionary takes the row_coordinates of the first line of the detected staffs, and spacing between the staffs, to assign symbols.
## It returns a dictionary with the row_number and the symbol.
def symbols_dictionary(row_coord, spacing):
    lookup_symbols = {}
    staf = 1
    offset = 2
    j = 0
    for i in row_coord:
        if (staf % 2 != 0):
            lookup_symbols[int(i)] = 'F'
            lookup_symbols[int(i + spacing * 0.5)] = 'E'
            lookup_symbols[int(i + spacing)] = 'D'
            lookup_symbols[int(i + spacing * 1.5)] = 'C'
            lookup_symbols[int(i + spacing * 2)] = 'B'
            lookup_symbols[int(i + spacing * 2.5)] = 'A'
            lookup_symbols[int(i + spacing * 3)] = 'G'
            lookup_symbols[int(i + spacing * 3.5)] = 'F'
            #lookup_symbols[int(i + spacing * 4)] = 'E'
            lookup_symbols[int(i + spacing * 4.5)] = 'D'
            lookup_symbols[int(i + spacing * 5)] = 'B'
            lookup_symbols[int(i - spacing * 0.5)] = 'G'
        else:
            lookup_symbols[int(i)] = 'A'
            lookup_symbols[int(i + spacing * 0.5)] = 'G'
            lookup_symbols[int(i + spacing)] = 'F'
            lookup_symbols[int(i + spacing * 1.5)] = 'E'
            lookup_symbols[int(i + spacing * 2)] = 'D'
            lookup_symbols[int(i + spacing * 2.5)] = 'C'
            lookup_symbols[int(i + spacing * 3)] = 'B'
            lookup_symbols[int(i + spacing * 3.5)] = 'G'
            lookup_symbols[int(i + spacing * 4)] = 'F'
            lookup_symbols[int(i + spacing * 4.5)] = 'E'
            lookup_symbols[int(i - spacing * 0.5)] = 'B'
            lookup_symbols[int(i - spacing)] = 'C'
            lookup_symbols[int(i - spacing * 2)] = 'B'
        staf += 1
    return lookup_symbols

## This function is used to remove the staff lines for the image and the templates. It is essentially used for noisy images, for better detections.
def remove_background_templates(template):
    h, w = template.shape
    for x in range(h):
        edge_cnt = 0
        for y in range(w):
            if (template[x][y] == 0):
                edge_cnt += 1
        # print(edge_cnt,w)
        if (edge_cnt > 0.7 * w):
            for y in range(w):
                template[x][y] = 1

    return template

## Implementation of sobel edge filter.
## Referred from ppt's taught in class.
def sobel(img):
    image_x = img.filter(ImageFilter.Kernel((3, 3), (-1, 0, 1, -2, 0, 2, -1, 0, 1), 1, 0))
    # image_x.show()

    image_y = img.filter(ImageFilter.Kernel((3, 3), (-1, -2, -1, 0, 0, 0, 1, 2, 1), 1, 0))
    # image_y.show()

    width, height = img.size

    image_sobel = np.zeros(shape=(height, width))

    for i in range(height):
        for j in range(width):
            x = image_x.getpixel((j, i))
            y = image_y.getpixel((j, i))
            image_sobel[i, j] = int(((x * x) + (y * y)) ** 0.5)

    image_sobel = Image.fromarray(image_sobel)
    # image_sobel.show()
    return image_sobel

## Implementation of canny edge filter.
## Referred from ppt's taught in class.
## Input -> pillow image, low threshold, high threshols
## output -> pillow image showing the edges
def canny(img, low_threshold, high_threshold):
    x_dir = [-1, 0, 1, 1, 1, 0, -1, -1]
    y_dir = [1, 1, 1, 0, -1, -1, -1, 0]
    image_sobel = sobel(img)

    width, height = image_sobel.size

    image_canny = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            x = image_sobel.getpixel((j, i))

            if (x >= high_threshold):
                image_canny[i, j] = 255
            elif (x >= low_threshold):
                image_canny[i, j] = 0
                for k in range(8):
                    if (i + k >= 0 and i + k < height and j + k >= 0 and j + k < width and image_sobel.getpixel(
                            (j + k, i + k)) > 0):
                        image_canny[i, j] = 255
            else:
                image_canny[i, j] = 0

            if (i == 0 or i == height - 1 or j == 0 or j == width - 1):
                image_canny[i, j] = 0

    for i in range(height):
        for j in range(width):
            if (image_canny[i, j] == 0):
                image_canny[i, j] = 255
            elif (image_canny[i, j] == 255):
                image_canny[i, j] = 0

    image_canny = Image.fromarray(image_canny)
    # print("Showing canny image")
    # image_canny.show()

    return image_canny

## References -> https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html#numpy.argwhere
## Input -> Numpy image
## Output -> Distance between two sets of lines, distance between lines, starting line co-ordinates of the staffs.
## Detected the lines having 50% of edges. Referred this technique from the below link.
## Reference -> https://github.com/ojaashampiholi/Optical_Music_Recognition/blob/main/Hough_Transform.py
def staff_detection(img):
    height, width = img.shape

    black_white_image = img

    staff_rows = []
    for i in range(height):
        if (len(np.argwhere(black_white_image[i] == 0)) >= 0.5 * width):
            staff_rows.append(i)

    black_img = np.zeros((img.shape))
    for i in range(len(staff_rows)):
        for j in range(width):
            black_img[staff_rows[i]][j] = 255

    staff_lines = []
    staff_space = []
    inter_space = []

    print("Printing staff rows")
    print(staff_rows)

    staff_ind = 0
    staff_space.append(staff_rows[1] - staff_rows[0])
    staff_lines.append(staff_rows[0])
    min_space = 1000

    skip_row = 0
    for row in range(len(staff_rows) - 1):
        # if (skip_row):
        #     skip_row = 0
        #     continue
        diff = staff_rows[row + 1] - staff_rows[row]
        if (diff > 50):
            # print("Printing row value for updating staff space",row)
            staff_lines.append(staff_rows[row + 1])
            staff_space.append(0)
            staff_ind += 1
        else:
            if (diff > staff_space[staff_ind] and diff < 30):
                # print("Hi")
                staff_space[staff_ind] = diff
                if (abs(staff_rows[row + 1] - staff_rows[row]) != 1):
                    min_space = min(min_space, diff)

    for i in range(len(staff_lines) - 1):
        inter_space.append(staff_lines[i + 1] - staff_lines[i])

    for i in range(len(staff_space)):
        staff_space[i] = min_space

    intra_space = min(staff_space)
    print("Number of stafs detected: ", len(staff_lines))
    print("Staff space: ",intra_space)
    return inter_space, intra_space, staff_lines

## Naive approach used for template matching
## It compares the cropped image and template to compute the confidence score.
def all_pixel_comparison(sub_image, template):
    x, y = template.shape
    count = 0
    for i in range(x):
        for j in range(y):
            if (sub_image[i][j] == template[i][j]):
                count = count + 1
    return count / (x * y)


## https://www.geeksforgeeks.org/python-pil-image-resize-method/
def resize_templates(space, template):
    return template.resize(
        (int(template.width * (space / template.height)), int(template.height * (space / template.height))))

## Cross_correlation technique for template matching. It compares canny edge images and the cropped image.
## This technique is referred from this link
## https://github.com/ojaashampiholi/Optical_Music_Recognition/blob/main/Template_Matching.py
def cross_correlation(sub_image, template):
    x, y = template.shape
    bbox_score = np.sum((sub_image * template) + (1 - sub_image) * (1 - template))
    return bbox_score / (x * y)

## Computes the edge score of the given crop of image. This basically returns the number of edges in the region passed.
def edge_score_roi(crop_img):
    return np.sum(crop_img)

## Function to draw bounding boxes and text for the detected contours.
## Reference -> https://www.geeksforgeeks.org/python-pil-imagedraw-draw-rectangle/
## Reference -> https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
def draw_bbox(contours, lookup_dict=None):
    image = Image.open("detected.png")
    image_shape = (np.array(image)).shape
    text_fill = (0,0,255)
    if(len(image_shape)<=2):
        text_fill = (0)
    color_arr = ["blue", "red", "green", "orange","orange","orange","orange"]
    index = -1
    template = 1
    symbols = []
    final_contour_ind = []
    for contour in contours:
        index += 1
        color = color_arr[index % len(color_arr)]
        clen = len(contour)
        for i in range(clen):
            img1 = ImageDraw.Draw(image)
            y_0, x_0, w, h, score = contour[i]
            x_1 = x_0 + int(h / 2)
            y_1 = y_0 + int(w / 2)

            if (template == 1):
                c_row = int((x_0 + x_1) / 2) + 2
                nearest_row = int((x_0 + x_1) / 2) + 2
                curr_diff = 6
                for key, val in lookup_dict.items():
                    diff = abs(c_row - key)
                    if (diff < curr_diff):
                        nearest_row = key
                        curr_diff = diff
                if (curr_diff >= 6):
                    continue
            img1.rectangle([(y_0, x_0), (y_0 + w, x_0 + h)], outline=color, width=3)

            if (template == 1):
                symbol = lookup_dict[nearest_row]
                myFont = ImageFont.load_default()
                # myFont = ImageFont.truetype('Helvetica', 60)
                img1.text((y_0 - 5, x_0 - 5), str(symbol), fill=text_fill, font=myFont)
                final_contour_ind.append(i)
                symbols.append(symbol)
        template += 1
    image.save('detected.png')
    return final_contour_ind, symbols

## Main function that does template matching.
## It implementes 2 techniques: naive and edge detection based on the arguments.
## Edge detection: Uses cross correlation and count of edge pixels of the template and the image for predicting the symbols.
## It returns the contours array: which contains: [y,x,w,h,confidence] values of each bbox detected.
def optical_music_recognition(image, template, canny_image, canny_template, row_number, spacing, cutoff=0.85,
                              comparision_type='naive', edge_thresh=10):
    count = 0
    confidence_score = []
    contours = []
    template_h, template_w = template.shape
    # print(template_h,template_w)
    image_h, image_w = image.shape
    for i in range(image_h - template_h):
        if (i < row_number - spacing * 3 or i > row_number + spacing * 7):
            continue
        for j in range(image_w - template_w):
            if (comparision_type == 'naive'):
                roi_img = image[i:i + template_h, j:j + template_w]
                score = all_pixel_comparison(roi_img, template)
                if (score > cutoff):
                    count = count + 1
                    contours.append([j, i, template_w, template_h, score])
                    # confidence_score.append(score)
            else:
                roi_img = canny_image[i:i + template_h, j:j + template_w]
                score = cross_correlation(roi_img, canny_template)
                tmp_edge_score = np.sum(canny_template)
                edge_score = edge_score_roi(roi_img)
                edge_score_diff = abs(edge_score - tmp_edge_score)

                if (score > cutoff and edge_score_diff < edge_thresh):
                    #print(score, cutoff, edge_score_diff)
                    count = count + 1
                    contours.append([j, i, template_w, template_h, score])
                    # confidence_score.append(score)
    return count, contours

## This function is used to eliminate all the overlapping bounding boxes that are detected.
## It takes in contours as the input and returns filtered contours.
def update_contours(contours):
    list_contours = list(contours)
    list_contours_copy = copy.deepcopy(list_contours)

    c_len = len(list_contours)
    i = 0

    for i in range(c_len):
        x, y, w, h, confi = list_contours[i]
        for j in range(i + 1, c_len):
            if (list_contours[j] not in list_contours_copy):
                continue
            if (abs(list_contours[j][0] - x) <= w * 0.75) and (abs(list_contours[j][1] - y) <= h * 0.75):
                list_contours_copy.remove(list_contours[j])

    return list_contours_copy

## This function is used to document the detections.
## Format is: [x,y,w,h,symbol_type,pitch,confidence]
def make_text_file(contours, template, symbols=None):
    cols = ['x', 'y', 'w', 'h', 'symbol_type', 'pitch', 'confidence']
    filename = (image_str.split('.'))[0]
    ind = -1
    with open('detected.txt', 'a') as file:
        for item_countours in contours:
            ind += 1
            item_score = item_countours[4]
            file.write(str(item_countours[1]) + " ") ## x
            file.write(str(item_countours[0]) + " ") ## y
            file.write(str(item_countours[2]) + " ") ## w
            file.write(str(item_countours[3]) + " ") ## h
            file.write(str(template) + " ")
            if template == "filled note":
                # print(symbols[ind])
                file.write(str(symbols[ind]) + " ")
            else:
                file.write("_" + " ")
            file.write(str(round(item_score, 2)) + "\n")
    file.close()

## This function is used to detect the extra symbols: ['treble clef','base clef','sharps','flats']
def optical_music_recognition_extraSymbols(image, template, canny_image, canny_template, row_number, spacing, cutoff,
                                     comparison_type='naive', factor=0.15):
    confidence_score = []
    count = 0
    contours = []
    template_h, template_w = template.shape
    image_h, image_w = image.shape
    max_score = 0

    for i in range(max(row_number - spacing * 3, 0), min(image_h - template_h, row_number + spacing * 7)):
        if (i < row_number - spacing * 3 or i > row_number + spacing * 7):
            continue
        for j in range((int)(factor * (image_w - template_w))):
            if (comparison_type == 'naive'):
                roi_img = image[i:i + template_h, j:j + template_w]
                score = all_pixel_comparison(roi_img, template)
                max_score = max(max_score, score)
                if (score >= cutoff):
                    count = count + 1
                    contours.append([j, i, template_w, template_h, score])
                    if (factor == 0.15):
                        return count, contours

    return count, contours


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        raise Exception(" python command is : python3 staff_finder.py input.jpg/input.png")

    image_str: str = sys.argv[1]

    if (image_str == "music1.png"):
        comparison_method = {'template1': 0.85, 'template2': 0.85, 'template3': 0.78, 'template4': 0.70,
                             'template5': 0.80, 'template6': 0.79, 'template7': 0.80, 'type1': 'naive',
                             'type2': 'naive', 'type3': 'naive', 'type4': 'naive', 'type5': 'naive', 'type6': 'naive',
                             'type7': 'naive', 'Gaussian_blur': False,'staff_removal':False,'edge_threshold': [10, 30, 50]}
    elif (image_str == "music2.png"):
        comparison_method = {'template1': 0.83, 'template2': 0.80, 'template3': 0.65, 'template4': 0.70,
                             'template5': 0.72, 'template6': 0.80, 'template7': 0.80, 'type1': 'naive', 'type2': 'edge',
                             'type3': 'edge', 'type4': 'naive', 'type5': 'naive', 'type6': 'naive', 'type7': 'naive',
                             'Gaussian_blur': False,'staff_removal':False,
                             'edge_threshold': [10, 30, 20]}
    elif (image_str == "music3.png"):
        comparison_method = {'template1': 0.8, 'template2': 0.75, 'template3': 0.65, 'template4': 0.70,
                             'template5': 0.73, 'template6': 0.80, 'template7': 0.80, 'type1': 'naive',
                             'type2': 'edge', 'type3': 'edge', 'type4': 'naive', 'type5': 'naive', 'type6': 'naive',
                             'type7': 'naive','Gaussian_blur': True,'staff_removal':False, 'edge_threshold': [10, 30, 20]}
    elif (image_str == "music4.png"):
        comparison_method = {'template1': 0.80, 'template2': 0.75, 'template3': 0.55, 'template4': 0.66,
                             'template5': 0.70, 'template6': 0.80, 'template7': 0.80, 'type1': 'naive',
                             'type2': 'edge', 'type3': 'edge', 'type4': 'naive', 'type5': 'naive', 'type6': 'naive',
                             'type7': 'naive','Gaussian_blur': False,'staff_removal':True,'edge_threshold': [10, 30, 30]}
    else:
        comparison_method = {'template1': 0.83, 'template2': 0.75, 'template3': 0.65, 'template4': 0.70,
                             'template5': 0.70, 'template6': 0.80, 'template7': 0.80, 'type1': 'naive',
                             'type2': 'edge', 'type3': 'edge', 'type4': 'naive', 'type5': 'naive', 'type6': 'naive',
                             'type7': 'naive', 'Gaussian_blur': False,'staff_removal':False,'edge_threshold': [10, 30, 30]}

    image = Image.open(image_str)
    image.save('detected.png')

    np_image_color = np.array(image)
    image_shape = np_image_color.shape
    np_image = np_image_color

    if (len(image_shape) > 2):
        np_image = 0.2989 * np_image_color[:, :, 0] + 0.5870 * np_image_color[:, :, 1] + 0.1140 * np_image_color[:, :,2]

    image_grayscale = Image.fromarray(np_image).convert('L')

    np_image = np.where(np_image < 128, 0, 255)

    inter_space, intra_space, line_begin = staff_detection(np_image)
    print("Printing intra space array", intra_space)

    template1 = Image.open('template1.png').convert("L")
    template2 = Image.open('template2.png').convert("L")
    template3 = Image.open('template3.png').convert("L")
    template4 = Image.open('template4.png').convert("L")
    template5 = Image.open('template5.png').convert("L")
    template6 = Image.open('template6.png').convert("L")
    template7 = Image.open('template7.png').convert("L")

    canny_temp_save = np.where(np.array(template2) < 128, 0, 1)
    canny_temp_save_2 = remove_background_templates(canny_temp_save)
    im = Image.fromarray((canny_temp_save_2 * 255).astype(np.uint8))
    im.save('t2_without_staff.png')

    canny_temp_save = np.where(np.array(template3) < 128, 0, 1)
    canny_temp_save_3 = remove_background_templates(canny_temp_save)
    im = Image.fromarray((canny_temp_save_3 * 255).astype(np.uint8))
    im.save('t3_without_staff.png')

    canny_image = np.array(canny(image_grayscale.convert('L'), 80, 200))
    canny_image = np.where(canny_image < 128, 0, 1)


    imageGB = Image.open(image_str)
    imageGB = imageGB.filter(ImageFilter.GaussianBlur(radius=2))
    #imageGB.save('GaussianBlur.png')
    np_image_colorGB = np.array(imageGB)

    if (len(image_shape) > 2):
        np_image_colorGB = 0.2989 * np_image_colorGB[:, :, 0] + 0.5870 * np_image_colorGB[:, :,
                                                                         1] + 0.1140 * np_image_colorGB[:, :, 2]

    image_grayscaleGB = Image.fromarray(np_image_colorGB).convert('L')

    np_imageGB = np.where(np_image_colorGB < 128, 0, 255)

    canny_image_GB = np.array(canny(image_grayscaleGB.convert('L'), 80, 200))
    canny_image_GB = np.where(canny_image_GB < 128, 0, 1)

    if (comparison_method['staff_removal']):
        canny_img_func = canny_image_GB.copy()
        im_2 = remove_background_templates(canny_img_func)
        canny_imageSR = np.where(im_2 < 128, 0, 1)
        im = Image.fromarray((im_2 * 255).astype(np.uint8))
        np_imageSR = np.where(im_2 <= 0, 0, 255)
        im.save('image_without_staff_{0}.png'.format(image_str.split('.')[0]))
        template2WS = Image.open('t2_without_staff.png').convert("L")
        template3WS = Image.open('t3_without_staff.png').convert("L")
        resize_template2WS = resize_templates(intra_space * 3, template2WS)
        resize_template3WS = resize_templates(intra_space * 2, template3WS)

        np_template2WS = np.array(resize_template2WS)
        np_template2WS = np.where(np_template2WS < 128, 0, 255)

        np_template3WS = np.array(resize_template3WS)
        np_template3WS = np.where(np_template3WS < 128, 0, 255)

        canny_template2WS = np.array(canny(resize_template2WS.convert('L'), 80, 200))
        canny_template2WS = np.where(canny_template2WS < 128, 0, 1)

        canny_template3WS = np.array(canny(resize_template3WS.convert('L'), 80, 200))
        canny_template3WS = np.where(np_template3WS <= 0, 0, 1)

    lookup_symbols = symbols_dictionary(line_begin, intra_space)

    contours_1_full = []
    contours_2_full = []
    contours_3_full = []
    contours_3WS_full = []
    contours_4_full = []
    contours_5_full = []
    contours_6_full = []
    contours_7_full = []

    symbols_arr = []

    resize_template1 = resize_templates(intra_space, template1)
    resize_template2 = resize_templates(intra_space * 3, template2)
    resize_template3 = resize_templates(intra_space * 2, template3)
    resize_template4 = resize_templates(intra_space * 8, template4)
    resize_template5 = resize_templates(intra_space * 3, template5)
    resize_template6 = resize_templates(intra_space * 3, template6)
    resize_template7 = resize_templates(intra_space * 3, template7)

    np_template1 = np.array(resize_template1)
    np_template1 = np.where(np_template1 < 128, 0, 255)

    np_template2 = np.array(resize_template2)
    np_template2 = np.where(np_template2 < 128, 0, 255)

    np_template3 = np.array(resize_template3)
    np_template3 = np.where(np_template3 < 128, 0, 255)

    np_template4 = np.array(resize_template4)
    np_template4 = np.where(np_template4 < 128, 0, 255)

    np_template5 = np.array(resize_template5)
    np_template5 = np.where(np_template5 < 128, 0, 255)

    np_template6 = np.array(resize_template6)
    np_template6 = np.where(np_template6 < 128, 0, 255)

    np_template7 = np.array(resize_template7)
    np_template7 = np.where(np_template7 < 128, 0, 255)

    canny_template1 = np.array(canny(resize_template1.convert('L'), 80, 200))
    canny_template1 = np.where(canny_template1 < 128, 0, 1)

    canny_template2 = np.array(canny(resize_template2.convert('L'), 80, 200))
    canny_template2 = np.where(canny_template2 < 128, 0, 1)

    canny_template3 = np.array(canny(resize_template3.convert('L'), 80, 200))
    canny_template3 = np.where(canny_template3 < 128, 0, 1)

    canny_template4 = np.array(canny(resize_template4.convert('L'), 80, 200))
    canny_template4 = np.where(canny_template4 < 128, 0, 1)

    canny_template5 = np.array(canny(resize_template5.convert('L'), 80, 200))
    canny_template5 = np.where(canny_template5 < 128, 0, 1)

    canny_template6 = np.array(canny(resize_template6.convert('L'), 80, 200))
    canny_template6 = np.where(canny_template6 < 128, 0, 1)

    canny_template7 = np.array(canny(resize_template7.convert('L'), 80, 200))
    canny_template7 = np.where(canny_template7 < 128, 0, 1)

    for row in range(len(line_begin)):
        row_number = line_begin[row]

        print("Detecting Template-1")

        contours_1 = []
        count_1_update, contours_1 = optical_music_recognition(np_image, np_template1, canny_image,
                                                               canny_template1, row_number, intra_space,
                                                               comparison_method['template1'],
                                                               comparison_method['type1'],
                                                               comparison_method['edge_threshold'][0])
        contours_1 = update_contours(contours_1)


        print("Detecting Template-2")

        imgfunc_canny = canny_image
        tempfunc2_canny = canny_template2
        tempfunc3_canny = canny_template3

        contours_2 = []
        count_2_update, contours_2 = optical_music_recognition(np_image, np_template2, imgfunc_canny,
                                                               tempfunc2_canny, row_number, intra_space,
                                                               comparison_method['template2'],
                                                               comparison_method['type2'],
                                                               comparison_method['edge_threshold'][1])
        contours_2 = update_contours(contours_2)


        print("Detecting Template-3")
        contours_3 = []

        if(comparison_method['Gaussian_blur']):
            count_3_update, contours_3 = optical_music_recognition(np_image, np_template3, canny_image_GB,
                                                                   tempfunc3_canny, row_number, intra_space,
                                                                   comparison_method['template3'],
                                                                   comparison_method['type3'],
                                                                   comparison_method['edge_threshold'][2])
        else:
            count_3_update, contours_3 = optical_music_recognition(np_image, np_template3, canny_image,
                                                                   tempfunc3_canny, row_number, intra_space,
                                                                   comparison_method['template3'],
                                                                   comparison_method['type3'],
                                                                   comparison_method['edge_threshold'][2])

        contours_3 = update_contours(contours_3)

        contours_3_WS = []
        if (comparison_method['staff_removal']):
            print('Detecting template 3 without stafs.')
            im = Image.fromarray((canny_template3WS * 255).astype(np.uint8))
            im.save('t3WS_check.png')

            count_3_update_WS, contours_3_WS = optical_music_recognition(np_image, np_template3, canny_image_GB,
                                                                         canny_template3WS, row_number, intra_space,
                                                                         0.8,
                                                                         comparison_method['type3'],
                                                                         comparison_method['edge_threshold'][2])

            contours_3_WS = update_contours(contours_3_WS)
            contours_3.extend(contours_3_WS)
        print('Detecting other symbols:')

        contours_4 = []
        count_4_update, contours_4 = optical_music_recognition_extraSymbols(np_image, np_template4, canny_image,
                                                                      canny_template4, row_number, intra_space,
                                                                      comparison_method['template4'])
        contours_4 = update_contours(contours_4)

        contours_5 = []
        count_5_update, contours_5 = optical_music_recognition_extraSymbols(np_image, np_template5, canny_image,
                                                                      canny_template5, row_number, intra_space,
                                                                      comparison_method['template5'])
        contours_5 = update_contours(contours_5)

        contours_6 = []
        count_6_update, contours_6 = optical_music_recognition_extraSymbols(np_image, np_template6, canny_image,
                                                                      canny_template6, row_number, intra_space,
                                                                      comparison_method['template6'],
                                                                      comparison_method['type6'], 1)
        contours_6 = update_contours(contours_6)

        contours_7 = []
        count_7_update, contours_7 = optical_music_recognition_extraSymbols(np_image, np_template7, canny_image,
                                                                      canny_template7, row_number, intra_space,
                                                                      comparison_method['template7'],
                                                                      comparison_method['type7'], 1)

        contours_7 = update_contours(contours_7)

        contours_templates_all = [contours_1, contours_2, contours_3, contours_4, contours_5, contours_6, contours_7]
        final_contours_I, symbols = draw_bbox(contours_templates_all, lookup_symbols)

        contours_1_upd = []
        for i in final_contours_I:
            contours_1_upd.append(contours_1[i])

        contours_1_full.extend(contours_1_upd)
        symbols_arr.extend(symbols)
        contours_2_full.extend(contours_2)
        contours_3_full.extend(contours_3)
        contours_4_full.extend(contours_4)
        contours_5_full.extend(contours_5)
        contours_6_full.extend(contours_6)
        contours_7_full.extend(contours_7)

    file = open('detected.txt', 'w')

    make_text_file(contours_1_full, 'filled note', symbols_arr)
    make_text_file(contours_2_full, 'eighth rest')
    make_text_file(contours_3_full, 'quarter rest')
    make_text_file(contours_4_full, 'treble clef')
    make_text_file(contours_5_full, 'base clef')
    make_text_file(contours_6_full, 'sharp')
    make_text_file(contours_7_full, 'flat')

