### 01.Importing libraries
from keras.models import Sequential
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # for image data augmentation gathering data from images

### 02.CNN Design
model = Sequential() #mentaining the layer oder of the model
model.add(Conv2D(32,(3, 3), input_shape=(64, 64, 3),activation='relu' )) # First layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu')) # Fully connected layer
model.add(Dense(units=1, activation='sigmoid')) # Output layer

model.compile( optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) ## Compiling the CNN

### 03.Pre-processing the data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Dataset/train',
                                                 target_size=(64, 64),
                                                 batch_size=8,
                                                 class_mode='binary')

validation_set = val_datagen.flow_from_directory('Dataset/val',
                                                 target_size=(64, 64),
                                                 batch_size=8,
                                                 class_mode='binary')

### 04.Training the CNN
model.fit(training_set,
                    steps_per_epoch=10,
                    epochs=50,
                    validation_data=validation_set,
                    validation_steps=2)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")
print("Saved model to disk")
