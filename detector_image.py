import matplotlib.pylab as plt
import cv2
import numpy as np

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
# we have to mask the rest lines(lets make a function for that)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img) # blank mask(matrix)
    channel_count = img.shape[2] #number of color channels
    match_mask_color = (255,) * channel_count # create a match color with same channel count
    cv2.fillPoly(mask, vertices, match_mask_color) # fill polygons to mask all other lines
    masked_image = cv2.bitwise_and(img, mask) # to take only the matching pixels
    return masked_image

cropped_image = region_of_interest(image,
                np.array([region_of_interest_vertices], np.int32),)

plt.imshow(cropped_image)
plt.show()