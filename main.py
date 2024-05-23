
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.dates as mdates
import datetime

# load data

ds, ds_info = tfds.load('cifar10', split=['train[:80%]', 'train[80%:90%]', 'test'], as_supervised=True, shuffle_files=True, with_info=True)

train_ds, val_ds, test_ds = ds

class_names = ds_info.features['label'].names
print("Class names:", class_names)

# preprocess an image
def preprocess_single_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# preprocess the dataset images
def preprocess(image, label):
    image = tf.image.resize(image, (64, 64))
    image = image / 255.0
    return image, label

# preprocess the train and test datasets

train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_ds = val_ds.map(preprocess).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



# NN architecture CIFAR10
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    # layers.Conv2D(256, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)), #l2 regularization
    layers.Dropout(0.5),
    # layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    # layers.Dropout(0.5),
    # layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    # layers.Dropout(0.5),
    # layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    # layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds
)


val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc}")

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc}")



#frog
#image_path ='/content/drive/MyDrive/cifar10/frog-2_ver_1.jpg'

#plane
#image_path ='/content/drive/MyDrive/cifar10/skynews-boeing-737-plane_5435020.jpg'

#truck
#image_path ='/content/drive/MyDrive/cifar10/Isuzu_elf400.jpg'

#bird
#image_path ='/content/drive/MyDrive/cifar10/indigo-bunting.jpg'

#horse
#image_path ='/content/drive/MyDrive/cifar10/horse.jpg'

#deer
image_path ='/content/drive/MyDrive/cifar10/deer.jpeg'

img_array = preprocess_single_image(image_path)


predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

print(f"Predicted class: {predicted_class[0]} ({class_names[predicted_class[0]]})")

# probabilities for each class
print("Probabilities for each class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {predictions[0][i]:.4f}")

# prediction
plt.imshow(load_img(image_path))
plt.title(f"Predicted: {class_names[predicted_class[0]]}")
plt.axis('off')
plt.show()


plt.figure(figsize=(12, 4))


#accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.annotate(f"{history.history['accuracy'][-1]:.2f}",
             (len(history.history['accuracy'])-1, history.history['accuracy'][-1]),
             textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate(f"{history.history['val_accuracy'][-1]:.2f}",
             (len(history.history['val_accuracy'])-1, history.history['val_accuracy'][-1]),
             textcoords="offset points", xytext=(0,10), ha='center')

#loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')


plt.annotate(f"{history.history['loss'][-1]:.2f}",
             (len(history.history['loss'])-1, history.history['loss'][-1]),
             textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate(f"{history.history['val_loss'][-1]:.2f}",
             (len(history.history['val_loss'])-1, history.history['val_loss'][-1]),
             textcoords="offset points", xytext=(0,10), ha='center')

plt.show()

