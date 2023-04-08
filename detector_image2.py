import matplotlib.pylab as plt
import cv2
import numpy as np

# we have to mask the rest lines(lets make a function for that)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img) # blank mask(matrix)
    #channel_count = img.shape[2] #number of color channels
    match_mask_color = 255 # create a match color with same channel count
    cv2.fillPoly(mask, vertices, match_mask_color) # fill polygons to mask all other lines
    masked_image = cv2.bitwise_and(img, mask) # to take only the matching pixels
    return masked_image

# function to draw the line
def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) #blank img with same dimmension
    for line in lines:
        for x1, y1, x2, y2 in line: # loop through lines
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0),thickness=10) # draw line
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

image = cv2.imread('road.jpg') # read the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert the image to rgb format

print(image.shape)
height = image.shape[0]
width = image.shape[1]

# Define Region of Tntereset
# the lane we are in has 2 paralal line merging at one point (a triangle)
region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]

# we need to take the grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# to detect the edge we will do canny edge detection
canny_image = cv2.Canny(gray_image, 100, 200)
# do region of interest again in canny image
cropped_image = region_of_interest(canny_image,
                np.array([region_of_interest_vertices], np.int32),)

# to draw the line in these edges using hough lines
lines = cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi/180,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=25)

image_with_lines = draw_lines(image, lines)

plt.imshow(image_with_lines)
plt.show()