# Histogram Calculation

In this section both histogram and cumulative histogram are calcualted for a given image. 

These histograms (normal and cumulative for the same image) are plotted in a single output image. 

The histogram images are of size 1024 in width and 512 in height, histograms are plotted using bars representation.

![alt text](https://github.com/abadrawy/HistogramsAndFilters/blob/master/images/bat.png)![alt text](https://github.com/abadrawy/HistogramsAndFilters/blob/master/histograms/batHis.png)



# Mean Versus Gaussian
Mean and Gaussian filters of the same size (5 x 5) are applied on the same image.

This step is followed by examining and comparing the results visually using the histograms of each.

Such a comparison shows the intensities redistribution caused by both filters.

The image subject of this test is the cameraman image.

# Selective Median Filter

Given an image with salt and peper noise, median filter is used to remove this noise.

Implemented the 5 x 5 median filter and ran it on the Fog and Noise image, and recorded the runtime of such a process.  
Then utilized the information provided by the histogram generated above in order to enhance the runtime of this process.
By only applying the filter on pixels that are most likely white or black as they are most likely to be the noise.
Thus, saving time, insted of applying it on all the pixels.

The result of applying a 5 x 5 median filter on the Fog and Noise image is as shown below. 

![alt text](https://github.com/abadrawy/HistogramsAndFilters/blob/master/images/fognoise.png)![alt text](https://github.com/abadrawy/HistogramsAndFilters/blob/master/images/fog.png)





# Contrast Stretching and Histogram Equalization
Implemented the techniques of contrast stretching and histogram equalization.

Both techniques are to be applied on the Fog image.

The output images scale utilizing the full range from 0 to 255.

Also, the histograms for both images is calculated.

# Libraries
opencv2

numpy

matplotlib
