# **Finding Lane Lines on the Road**

## **Objectives summary**
For an overall summary, this project takes a video pipeline and processes the images to show how the algorithm can define lane line boundaries via a python video pipeline of multiple filters.

## Install

The following plugins need to be installed in cmd for this program to work:
pip (if not already available) for plugin downloads via cmd, numpy for array operations, opencv for working with videos and images, ffmpeg for optional codec for compression if needed

```
python get-pip.py
git clone https://github.com/numpy/numpy.git numpy
pip install opencv-python
pip install ffmpeg-normalize
```

## **Algorithm summary**

For the technical detail summary, this project imports a video from a "\\test_videos" location and captures frame by frame for processing via multiple filters. 


Initially, a grayscale filter is applied so the image can be more easily processed as 1-channel and more easily defines objects by a 1 dimensional hue difference.

```
def grayscale(img):
    #Applies the Grayscale transform
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```
Next, a gaussian blur filter is applied to remove some noise. Afterwards, a canny edge filter is applied to detect edges based on gradient change and threshold parameters. 

```
def gaussian_blur(img, kernel_size):
    #Applies a Gaussian Noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
Then a canny edge filter is applied to find changes in pixel intensity, effectively defining the line lanes based on a set threshold.

```
def canny(img, low_threshold, high_threshold):
    #Applies the Canny transform
    return cv2.Canny(img, low_threshold, high_threshold)
```


After this, a mask is applied to define which region will be of interest to detect, in this case only the lane lines and partial noise depending on obstacles. 

```
def region_of_interest(img, vertices):
	#returns polygonal mask
	
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

The final filter is Hough transform, which draws lines in a specified color (red in this case) via multiple parameters of straight lines found on the image. 

```
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, height):
    #Returns an image with hough lines drawn.
	
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, height)
    return line_img
```


The draw_lines() function was modified to find the beginning and end points of the detected line boundaries and afterwards averaging them out to draw the lines in the middle of the boundaries. After this, a blank image with these lines is combined with the original image via weight_img function. This image is then compressed and saved into a video pipeline in folder "\\testdir" and the image is removed. This is done until all the files in the directory are processed. 

```
def draw_lines(img, lines, height, color=[0, 0, 255], thickness=10):
    #draws lines over the image
	
    top = int(height*7/12)#320
    bottom = int(height*10/12)#550
    left_x1s = []
    left_y1s = []
    left_x2s = []
    left_y2s = []
    right_x1s = []
    right_y1s = []
    right_x2s = []
    right_y2s = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            # Draw line segments in blue for error checking.
            cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 6)
            
            slope = get_slope(x1,y1,x2,y2)
            if slope < 0:
                # Ignore obviously invalid lines
                if slope > -.5 or slope < -.8:
                    continue        
                left_x1s.append(x1)
                left_y1s.append(y1)
                left_x2s.append(x2)
                left_y2s.append(y2)
            else:
                # Ignore obviously invalid lines
                if slope < .5 or slope > .8:
                    continue        
                right_x1s.append(x1)
                right_y1s.append(y1)
                right_x2s.append(x2)
                right_y2s.append(y2)
                
    try:
        avg_right_x1 = int(np.mean(right_x1s))
        avg_right_y1 = int(np.mean(right_y1s))
        avg_right_x2 = int(np.mean(right_x2s))
        avg_right_y2 = int(np.mean(right_y2s))
        right_slope = get_slope(avg_right_x1,avg_right_y1,avg_right_x2,avg_right_y2)

        right_y1 = top
        right_x1 = int(avg_right_x1 + (right_y1 - avg_right_y1) / right_slope)
        right_y2 = bottom
        right_x2 = int(avg_right_x1 + (right_y2 - avg_right_y1) / right_slope)
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    except ValueError:
        # Don't error when a line cannot be drawn
        pass

    try:
        avg_left_x1 = int(np.mean(left_x1s))
        avg_left_y1 = int(np.mean(left_y1s))
        avg_left_x2 = int(np.mean(left_x2s))
        avg_left_y2 = int(np.mean(left_y2s))
        left_slope = get_slope(avg_left_x1,avg_left_y1,avg_left_x2,avg_left_y2)

        left_y1 = top
        left_x1 = int(avg_left_x1 + (left_y1 - avg_left_y1) / left_slope)
        left_y2 = bottom
        left_x2 = int(avg_left_x1 + (left_y2 - avg_left_y1) / left_slope)
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)        
    except ValueError:
        # Don't error when a line cannot be drawn
        pass

def get_slope(x1,y1,x2,y2):
	#slope for 4 input values
	return ((y2-y1)/(x2-x1))

def weighted_img(img, initial_img, alpha=0.8, beta=1., lambd=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * (alpha) + img * (beta) + (lambd)
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambd)
```

Here is an image that went through the algorithm, showing all that was mentioned.

![Frame 0 of Challenge](https://github.com/Olexyv/Project-1/blob/master/frame0.jpg)

## **Possible improvements and limitations**

The algorithm itself could improve by adding more filters that will reduce noise in the image. In a lot of the cases the drawn lane lines flicker which is telling that there are missing points at the extremas, large interpolated slopes (which are defined out of bounds in the algorithm) and disconnected lines. As stated before, filters could reduce the amount of points to be worked on, improving the slopes of the lines. Cluster averaging similar to Voronoi diagram points, or any nearest neighbor alternative, may help in averaging the cluster of points to a single point, effectively reducing the noise. Prediction of line continuity based on past slopes can remove the flickering problem altogether. Adding that to a cluster averaging algorithm (if that doesn't work, a predictive line draw based on past information) will draw the lane inline with the drawn line without jittering.
Looking at more large scale improvements, of course the algorithm would be significantly better if the road itself was identified without the need of a virtually shaped mask, (since its boundaries more or less helped the algorithm stay in a controlled system) as well as passing by objects and cars. Making some of the static parameters of the algorithm more dynamic with if statements for additional processing will help the algorithm as well, though will also increase the time taken for completion.
Possible bug was found that if the process runs too fast or runs into some timing issue, os.remove may run faster than cv2.VideoWriter and may possibly remove the image file before it is written, in which case the code can easily be modified with a process check or wait check. Overall, the algorithm does its function fairly well and can only get more robust with better filtering and identification methods.
