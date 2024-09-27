import cv2
import keras
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical_1.h5')

image = cv2.imread('D:\\MRI\\uploads\\FILE39_2.jpg') 

img = Image.fromarray(image)

img = img.resize((64, 64))  # Resize the image

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# Thay thế predict_classes bằng predict và argmax
result = model.predict(input_img)
predicted_class = np.argmax(result, axis=1)

print(predicted_class)
               




