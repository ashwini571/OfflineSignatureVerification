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
import math
import numpy as np
from scipy import ndimage


image = cv2.imread('cedar_dataset/full_forg/forgeries_2_4.png',cv2.IMREAD_GRAYSCALE)
image = cv2.imread('cedar_dataset/full_org/original_7_17.png',cv2.IMREAD_GRAYSCALE)
image = cv2.imread('cedar_dataset/full_org/original_22_16.png',cv2.IMREAD_GRAYSCALE)

image = cv2.imread('my_sig.jpg',cv2.IMREAD_GRAYSCALE)
image = cv2.imread('rotated.png',cv2.IMREAD_GRAYSCALE)



#rotation angle in degree

           

res = preprocess(image)
cv2.imshow('r',res)



def preprocess(image):
    clean = remove_background(image)
    roi = extract_signature(clean)
    th,res= cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # r = ndimage.rotate(res, -(90-math.degrees(math.atan(res.shape[1]/res.shape[0]))))
    angle = 90-math.degrees(math.atan(res.shape[1]/res.shape[0]))
    if angle > 20:
        res = rotate_image(res, -(angle if angle>20 else 0))
   
    return res

def rotate_image(image, angle):
  rot_mat = cv2.getRotationMatrix2D((image.shape[0]/2+40,image.shape[1]/2+40), angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, (int(math.sqrt(image.shape[0]*image.shape[0]+image.shape[1]*image.shape[1])),int(image.shape[1])), flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
  return result



def extract_signature(image):
    result = image.copy()
    img = image.copy()
    ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cnts,hi = cv2.findContours(closing.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x,y, x+w,y+h])
    
    boxes = np.asarray(boxes)
    left = np.min(boxes[:,0])
    top = np.min(boxes[:,1])
    right = np.max(boxes[:,2])
    bottom = np.max(boxes[:,3])
    
    result[closing==0] = (255)
    ROI = result[top:bottom, left:right].copy()
    cv2.rectangle(result, (left,top), (right,bottom), (36, 255, 12), 2)
    
    # cv2.imshow('result', result)
    # cv2.imshow('ROI', ROI)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('closing',closing)

    cv2.waitKey()
    
    return ROI


def remove_background(img):
        """ Remove noise using OTSU's method.

        :param img: The image to be processed
        :return: The normalized image
        """
        img = img.astype(np.uint8)
        # Binarize the image using OTSU's algorithm. This is used to find the center
        # of mass of the image, and find the threshold to remove background noise
    
        threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Remove noise - anything higher than the threshold. Note that the image is still grayscale
        img[img > threshold] = 255

        return img


