
# Detecting symbols and pitch of music notes

### Input Image
![alt text](https://github.com/tanmayiballa/Optical_music_recognition/blob/main/results/music1.png)

### Preprocessing:

#### Gaussian Blur:
Gaussian blur is used to smoothen the noisy images. 

#### Removing staff lines:
Staff lines present in music notes often tend to reduce the accuracy of detections, specially when we are performing edge based template matching.
Hence, we choose to remove staffs from noisy images, and perform gaussian blurring, before detecting symbols.
The staff lines in the templates are also removed for precise predictions. 

Below is the result of applying staff_removal and gaussian smoothing for the input image.

![alt text](https://github.com/tanmayiballa/Optical_music_recognition/blob/main/results/image_without_staff.png)

Staff removal performed on template-2 and template-3.
![alt text](https://github.com/tanmayiballa/Optical_music_recognition/blob/main/results/t2_without_staff.png)
![alt text](https://github.com/tanmayiballa/Optical_music_recognition/blob/main/results/t3_without_staff.png)

Another result of Gaussian blur and staff removal for noisy images : Music4.png
![alt_text](https://github.com/tanmayiballa/Optical_music_recognition/blob/main/results/image_without_staff_music4.png)

The above pre-processing is done only for noisy images. They seemed to improve the number of true positives detected. However, there are lot more false positives that are also detected. A detailed explanation of the same is given in the difficulties faced section. 

### Algorithms for template matching

We have used different techniques for matching templates.

## Pixel-to-pixel comparison

- In this technique, the pixels of the cropped image and the template are compared for each pixel, and the percentage of the pixels matched is used as the bounding box score. 
- A cutoff is applied to filter the same. This is implemented in the `all_pixel_comparison()` function. 
- This direct matching seems to work well for images without noise. Hence, this type of matching is used to detect template-1,2,and 3 for music1.png. 
- From experimentation, we have also observed that the performace of template-1 using naive matching was better than the other techniques. Hence, we have restricted the techique for template-1 as naive.

### Challenges faced:
 - Detections using naive method are not proper for images with noise.
 - The thresholds for the templates should be set to a very low value to predict the true positives. However, because of low thresholding, we are facing a lot of false positive detections.
 - Not time efficient. Since, we have to do a pixel-to-pixel comparison, this technique takes much time to process. Specially in cases where the image has many stafss (rach.png).

## Cross-correlation using edge image

- In this technique, we first computed the edge image of the templates and the input image using `canny()` and `sobel()` functions. 
- These edge images are used to find the cross correlation score of the image patch, using the formula: `image_patch*template + (1-image_patch)*(1-template)`. 
- This has resulted in better detections. However, we still faced the issue of false positives.
- In order to reduce the false positives, we have used another score for the difference in the edge count of the template and the image patch. 
- Formula for edge_score_diff: abs(count_edges(template) - count_edges(image_patch))<=x. The value of `x` and `cross_correlation_score` are fixed based on experimentation.

- All the thresholds are set based on experimentation.

### Challenges faced:
- False positives are significantly high. This is because of the small template sizes. Based on our observations, most of the false positives have similar structure of the templates.
- Cross correlation is easily manipulated by noisy images. Since, we only have the edge information, any other noisy edge in the image patch, can disrupt the score.
- Edge score that is used to detect the false positives, ignores the spatial location of the pixels in the image patch, i.e., since we are only comparing the count of the edges in the image, any other noisy edge in the image patch can be considered as true edge.

#### Usage

`omr.py` is the main file that has all the functions included. The function `optical_music_recognition()` implements the algorithms discussed above.
Also, we have limited the search to -3*spacing < line_begin < 7*spacing, to limit the false positives. Most of the symbols are concentrated in this region.

The optical_music_recognition() runs for each of the detected staff. In this way, we are trying to limit the algorithm search within the region of each staff. 

To test the input, use: `python omr.py <input_image>`

#### Extra Symbols detection:

Apart from the templates provided, we have also detected extra symbols: [treble clef, base clef, sharps, flats]. We believe that these symbols, especially treble_clef, and base_clef have significant importance in music notes. Hence, we choose to detect the same. All these extra symbols are displayed in orange, in the resultant image. The details of the same are also added in the detected.txt file. This symbol detection works perfectly for music1.png. 

##### Challenges faced:
- The template 5 is not of the same size in every image. Hence, we took a factor of 3 which is the avg of the sizes observed in the images at hand to resize the templates. 
- Few false positives for music-2,3, and 4.

#### Results
The detected bounding boxes are displayed in the input image using PIL library. The pitch of filled notes are also detected using `symbols_dictionary()`. 
Additionally, all the detected bounding boxes and the template information is written to detected.txt.

Music - 1 detections:
![alt_text](https://github.com/tanmayiballa/Optical_music_recognition/blob/main/results/music1_detected.png)

detected.txt contains lines in the given format: `['x', 'y', 'w', 'h', 'symbol_type', 'pitch', 'confidence']`

If the input image is a grayscale image (rach.png), the symbols get displayed in black, and the bounding boxes are displayed in gray. These detections might not be properly visible in the output image. However, the details of the detected symbols along with their bounding boxes are written in the detected.txt file.

#### Other Techniques tried
- Pixel to Pixel comparison is a very slow process. Additionally, we are using raw pixels as data which is not the ideal way to do any task in computer vision. In this regard we have explored template matching by Fourier transform but could not get proper results in the given time. The PDF of the paper is attached in the comments of the assignments.
- Though feature matching techniques such as, SIFT, ORB, SURF, etc. are used for image matching and registration, they doesn't increase accuracy beyond a limit. 
- Even with the methods implemented in our algorithm, a further fine tuning and experimentation, could give better results, and eliminate false positives, but it would be difficult to match the efficiency of CNNs.
- In this era of deep-learning, traditional feature matching techniques are not used widely. Since Convolution neural networks(CNNs) capture majority of the low-level features, they perform much better in template matching. 
