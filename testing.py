from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

def classify(img_path):
    img_name=img_path
    img = image.load_img(img_name, target_size = (64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    pred = model.predict(img)
    if pred[0][0] ==1:
        predivtion="Vadivelu"
    else:
        predivtion="Sangakkara"
    print(predivtion,img_name)
    
     
import os
path='Dataset/test'
files =[]
#r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
          files.append(os.path.join(r, file))
for f in files:
    classify(f)
    print('\n')