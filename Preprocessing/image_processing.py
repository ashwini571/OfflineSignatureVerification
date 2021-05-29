# Gray Scale -- done
# Noise Removal from background -- done
# Extract sign -- done
    # Thresholding
    # Morphological Transformation 
# Rotation -- done
# # Normalization(we normalize each image by dividing
# the pixel values with the standard deviation of the pixel
# values of the images in a dataset.)
# Resizing


import cv2
import numpy as np
from scipy import ndimage
import math


image = cv2.imread('Test/ash_org_1.jpg',cv2.IMREAD_GRAYSCALE)
image = cv2.imread('Test/rotated.png',cv2.IMREAD_GRAYSCALE)
           

img1 = cv2.imread('Datasets/cedar_dataset/full_org/original_5_1.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Datasets/cedar_dataset/full_org/original_23_15.png',cv2.IMREAD_GRAYSCALE)


k = preprocess(image)
cv2.imshow('a',res)

# r = ndimage.rotate(image, -(90-math.degrees(math.atan(image.shape[1]/image.shape[0]))))
# angle = 90-math.degrees(math.atan(image.shape[1]/image.shape[0]))

# res = rotate_image(image, -(angle if angle>20 else 0))



# def rotate_image(image, angle):
#   rot_mat = cv2.getRotationMatrix2D((image.shape[0]/2+40,image.shape[1]/2+40), angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, (int(math.sqrt(image.shape[0]*image.shape[0]+image.shape[1]*image.shape[1])),int(image.shape[1])), flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
#   return result

def preprocess(img, canvas_size=(700, 1070), img_size=(170, 242), input_size=(155, 220)):
    img_w, img_h = canvas_size
    # Background removal
    th, bin_image = remove_background(img)
    # cv2.imshow('Background removal', bin_image)
    
    # Extracting signature from image
    x_start, y_start, img_x, img_y, cropped = extract_signature(img, bin_image, img_w, img_h)
    cv2.imshow('Extracted', cropped)
    canvas_centered = canvas_centering(x_start, y_start, img_x, img_y, cropped,img_w, img_h, th)
    cv2.imshow('Canvas Centered', canvas_centered)
    inverted = 255 - canvas_centered
    resized = resize_image(inverted, img_size)
    cropped = crop_center(resized, input_size)
    
    for i in range(len(cropped)):
        for j in range(len(cropped[0])):
            if cropped[i][j] == 0:
                cropped[i][j]=255
            else:
                cropped[i][j] = 0
        
    return cropped 

    


def canvas_centering(x_start, y_start, img_x, img_y, cropped,img_w, img_h, th):
    canvas_image = np.ones((img_w, img_h), dtype=np.uint8) * 255
    # Add the image to the blank canvas
    canvas_image[x_start:x_start + img_x, y_start:y_start + img_y] = cropped
    cv2.imshow('Canvas with noise', canvas_image)
    # Remove noise - anything higher than the threshold. Note that the image is still grayscale
    canvas_image[canvas_image > th] = 255

    return canvas_image


def remove_background(img):
    """ Remove noise using gaussian filter and OTSU's method.
    """

    # Gaussian filter for removing small components
    blur_radius = 2
    blurred_image = ndimage.gaussian_filter(img, blur_radius)
    # cv2.imshow('Blurred',blurred_image)
    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
    th, bin_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return th, bin_image

def extract_signature(img, bin_image,img_w,img_h):
    # Find the center of mass
    x, y = np.where(bin_image == 0)
    x_center = int(x.mean() - x.min())
    y_center = int(y.mean() - y.min())

    # Crop the image with a tight box
    cropped = img[x.min(): x.max(), y.min(): y.max()]
    # cv2.imshow('cropped', cropped)
    
    # Center the image
    img_x, img_y = cropped.shape

    x_start = img_w // 2 - x_center
    y_start = img_h // 2 - y_center
    extra_x = (x_start + img_x) - img_w
    if extra_x > 0:
        x_start -= extra_x
    if x_start < 0:
        x_start = 0
    extra_y = (y_start + img_y) - img_h
    if extra_y > 0:
        y_start -= extra_y
    if y_start < 0:
        y_start = 0
        
    
    return x_start, y_start, img_x, img_y, cropped
        
    

def resize_image(image, new_size):
    height, width = new_size

    # Check which dimension needs to be cropped
    # (assuming the new height-width ratio may not match the original size)
    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(image.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(image.shape[0] / width_ratio))

    # Resize the image 
    image = cv2.resize(image.astype(np.float32), (resize_width,resize_height), interpolation=cv2.INTER_LINEAR)

    # Crop to exactly the desired new_size, using the middle of the image:
    if width_ratio > height_ratio:
        start = int(round((resize_width-width)/2.0))
        return image[:, start:start+width]
    else:
        start = int(round((resize_height-height)/2.0))
        return image[start:start+height, :]


def crop_center(img, input_shape):
    img_shape = img.shape
    start_y = (img_shape[0] - input_shape[0]) // 2
    start_x = (img_shape[1] - input_shape[1]) // 2
    cropped = img[start_y: start_y + input_shape[0], start_x:start_x + input_shape[1]]
    return cropped



