import os
import time
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib

start = time.time()

pinfo = pd.read_csv('info.csv')
path = 'C:/Users/bruno/OneDrive/Bilder/scaled_images_interlinear'
bilder = os.listdir(path)

train_images = []
train_ages = []
test_images = []
test_ages = []
i = 0

while i < len(path)/2:
    tmp = path+'/{Bild}'.format(Bild=bilder[i])
    train_images.append(nib.load(path+'/{Bild}'.format(Bild=bilder[i])))
    survey_txt = tmp.find('-s') + 2
    survey = int(tmp[survey_txt])
    match survey:
        case 0:
            train_ages.append(pinfo.iloc[i, 4])
        case 2:
            train_ages.append(pinfo.iloc[i, 3])
        case 4:
            train_ages.append(pinfo.iloc[i, 7])
        case 6:
            train_ages.append(pinfo.iloc[i, 8])
i = i + 1
print('Trainingsbilder geladen')

while i < len(path):
    tmp = path+'/{Bild}'.format(Bild=bilder[i])
    test_images.append(nib.load(path+'/{Bild}'.format(Bild=bilder[i])))
    survey_txt = tmp.find('-s') + 2
    survey = int(tmp[survey_txt])
    match survey:
        case 0:
            test_ages.append(pinfo.iloc[i, 4])
        case 2:
            test_ages.append(pinfo.iloc[i, 3])
        case 4:
            test_ages.append(pinfo.iloc[i, 7])
        case 6:
            test_ages.append(pinfo.iloc[i, 8])
i = i + 1
print('Testbilder geladen')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_ages, epochs=10,
                    validation_data=(test_images, test_ages))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_ages, verbose=2)

print(test_acc)

end = time.time()
length = end - start
print("Prozessdauer: ",length)