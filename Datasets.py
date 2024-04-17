import PIL
import cv2
import numpy as np
import random

x_training_data = []
y_training_data = []
for i in range(281):
    try:
        cropx = PIL.Image.open('file_path'.format(i))
        crop_arrayx = np.array(cropx)
        random_number = random.randint(2, 6)
        n = 2*random_number - 1
        ksize = (n,n)
        sigmaX = 0
        blurred_image = cv2.GaussianBlur(crop_arrayx, ksize, sigmaX)
        
        
        img2 = cv2.imread('file_path'.format(i))
        b, g, r = cv2.split(img2)

        ret, thf = cv2.threshold(r, 254, 255, cv2.THRESH_BINARY)
        ret, thm = cv2.threshold(g, 254, 255, cv2.THRESH_BINARY)
        ret, thv = cv2.threshold(b, 254, 255, cv2.THRESH_BINARY)
        channels = [thf, thm, thv]
        img = np.array(channels)
        cropy = img.transpose(1, 2, 0)
        
        rotation_number = random.randint(1, 3)
        if rotation_number == 1:
            rotated_image = cv2.rotate(blurred_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image_label = cv2.rotate(cropy, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_number == 2:
            rotated_image = cv2.rotate(blurred_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image_label = cv2.rotate(cropy, cv2.ROTATE_90_CLOCKWISE)
            rotated_image_label = cv2.rotate(rotated_image_label, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_number == 3:
            rotated_image = cv2.rotate(blurred_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image_label = cv2.rotate(cropy, cv2.ROTATE_90_CLOCKWISE)
            rotated_image_label = cv2.rotate(rotated_image_label, cv2.ROTATE_90_CLOCKWISE)
            rotated_image_label = cv2.rotate(rotated_image_label, cv2.ROTATE_90_CLOCKWISE)
            
        
            
        x_training_data.append(crop_arrayx)
        x_training_data.append(rotated_image)


        y_training_data.append(cropy)
        y_training_data.append(rotated_image_label)

    except:
        pass
  






x_val_data = []
y_val_data = []
for i in range(30):
    n_val = random.randint(1,238)
    x_val_data.append(x_training_data.pop(n_val))
    y_val_data.append(y_training_data.pop(n_val))
    
    
    

x_train = np.array(x_training_data, dtype=object)
x_train = x_train.astype("float32")/255
x_train = x_train.reshape(x_train.shape + (1,))

x_val = np.array(x_val_data, dtype=object)
x_val = x_val.astype("float32")/255
x_val = x_val.reshape(x_val.shape + (1,))


y_train = np.array(y_training_data, dtype=object)
y_train = y_train.astype("float32")/255

y_val = np.array(y_val_data, dtype=object)
y_val = y_val.astype("float32")/255

