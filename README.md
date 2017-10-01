#**Finding Lane Lines on the Road**

###**Finding Lane Lines on the Road**###

####**Objectives summary**####
For an overall summary, this project takes a video pipeline and processes the images to show how the algorithm can define lane line boundaries via a python video pipeline of multiple filters.

####**Algorithm summary**####
For the technical detail summary, this project imports a video from a "\\test_videos" location and captures frame by frame for processing via multiple filters. 
Initially, a grayscale filter is applied so the image can be more easily processed as 1-channel and more easily defines objects by a 1 dimensional hue difference. Next, a gaussian blur filter is applied to remove some noise. Afterwards, a canny edge filter is applied to detect edges based on gradient change and threshold parameters. After this, a mask is applied to define which region will be of interest to detect, in this case only the lane lines and partial noise depending on obstacles. The final filter is Hough transform, which draws lines in a specified color (red in this case) via multiple parameters of straight lines found on the image. The draw_lines() function was modified to find the beginning and end points of the detected line boundaries and afterwards averaging them out to draw the lines in the middle of the boundaries. After this, a blank image with these lines is combined with the original image. This image is then saved into a video pipeline in folder "\\testdir" and the image is removed. This is done until all the files in the directory are processed. Here is an image that went through the algorithm, showing all that was mentioned.

IMAGES NEEDED.

####**Possible improvements and limitations**####

The algorithm itself could improve by adding more filters that will reduce noise in the image. In a lot of the cases the drawn lane lines flicker which is telling that there are missing points at the extremas, large interpolated slopes (which are defined out of bounds in the algorithm) and disconnected lines. As stated before, filters could reduce the amount of points to be worked on, improving the slopes of the lines. Cluster averaging similar to Voronoi diagram points, or any nearest neighbor alternative, may help in averaging the cluster of points to a single point, effectively reducing the noise. Prediction of line continuity based on past slopes can remove the flickering problem altogether. Adding that to a cluster averaging algorithm (if that doesn't work, a predictive line draw based on past information) will draw the lane inline with the drawn line without jittering.
Looking at more large scale improvements, of course the algorithm would be significantly better if the road itself was identified without the need of a virtually shaped mask, (since its boundaries more or less helped the algorithm stay in a controlled system) as well as passing by objects and cars. Making some of the static parameters of the algorithm more dynamic with if statements for additional processing will help the algorithm as well, though will also increase the time taken for completion.
