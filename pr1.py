import numpy as np
import cv2
import os

#-------------------------Function definitions

def grayscale(img):
    #Applies the Grayscale transform
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
def canny(img, low_threshold, high_threshold):
    #Applies the Canny transform
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    #Applies a Gaussian Noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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
		
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, height):
    #Returns an image with hough lines drawn.
	
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, height)
    return line_img

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
	
#-------------------------------End of function definitions

#>>>Main algorithm

def process_image():
	#Initialize folders, filenames and codecs

	#Create the folders (if not already existing) to move the images/videos to
	pathsrc="test_videos\\"

	pathdest="testdir\\"
	if not os.path.exists(pathdest):
		os.makedirs(pathdest)

	#Change \\ to / ; useful for long path names that are copy pasted without changing
	pathsrc = '/'.join(pathsrc.split('\\'))
	pathdest = '/'.join(pathdest.split('\\'))

	#puts files names into array
	pathlist=os.listdir(pathsrc)
	files=[]
	for i in range (0 , len(pathlist)):
		files.append(os.path.join(pathsrc,pathlist[i]))

	outfiles=[]
	for i in range (0 , len(pathlist)):
		outfiles.append(os.path.join(pathdest,pathlist[i]))

	fourcc=0x00000021 #default code for X264 codec

	prename=pathdest + "frame"
	extension=".jpg"
	compression_parameter=[cv2.IMWRITE_JPEG_QUALITY, 50] #compression range 1-100; 100 = no compression

    # Processing image/video pipeline
	i=0 #counter for tracking variables
	c=0 #counter for cap
	
	for file in files:
	
		cap = cv2.VideoCapture(files[c])
		width=int(cap.get(3))
		height=int(cap.get(4))
		fps=int(cap.get(5))
		print("Working on file " +files[c])
		video = cv2.VideoWriter(outfiles[c],fourcc, fps, (width,height))
		
		while cap.isOpened():
			#read in image: boolean ret if frame null. image is the read image from cv2.VideoCapture
			ret,image = cap.read()
			
			if not ret: #checks if frame not null, removes error (-215)
				i=0 #reset counter for next image
				c+=1
				
				break
			
			gray = grayscale(image)
			
			# Define a kernel size and apply Gaussian smoothing
			kernel_size = 5
			blur_gray = gaussian_blur(gray, kernel_size)

			# Define our parameters for Canny and apply
			low_threshold = 50
			high_threshold = 150
			edges = canny(blur_gray, low_threshold, high_threshold)

			# Next we'll create a masked edges image using cv2.fillPoly()
			# This time we are defining a four sided polygon to mask
			imshape = image.shape
			vertices = np.array([[(0,imshape[0]),(imshape[1]*5/12, imshape[0]*9/12), (imshape[1]*7/12, imshape[0]*9/12), (imshape[1],imshape[0])]], dtype=np.int32)

			masked_edges=region_of_interest(edges,vertices)

			# Define the Hough transform parameters
			# Make a blank the same size as our image to draw on
			rho = 1 # distance resolution in pixels of the Hough grid
			theta = np.pi/180 # angular resolution in radians of the Hough grid
			threshold = 15     # minimum number of votes (intersections in Hough grid cell)
			min_line_length = 40 #minimum number of pixels making up a line
			max_line_gap = 20    # maximum gap in pixels between connectable line segments

			# Iterate over the output "lines" and draw lines on a blank image (as part of hough_lines function)

			# Run Hough on edge detected image
			# Output "lines" is an array containing endpoints of detected line segments
										
			lines = hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap, height)			
			
			# Draw the lines on the edge image
			lines_edges = cv2.addWeighted(
			image #color_edges
			, 0.8, lines, 1, 0) 
			
			#makes name for processed picture file, compresses and moves to directory
			name=prename+str(i)+extension
			cv2.imwrite(name, lines_edges, compression_parameter) #to get no compression, remove compression_ parameter
			
			#AFTER processing, replace frame with processed image
			
			a=cv2.imread(name)
			cv2.imshow('window-name',a) # Shows images for testing
			video.write(a)
			#Optional wait process may be needed to put here if file is removed before video write
			os.remove(name) #deletes image file, only keeps video
			i += 1

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
			
			i=i+1
				
process_image()

