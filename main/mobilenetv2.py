import tensorflow as tf
import numpy as np
import PIL
import random

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
print("Done")

random.seed(0)
np.random.seed(0)
tf.random.set_seed(42)
tf.random.set_seed(42)

BATCH_SIZE = 32
IMG_SIZE = (160,160)

train_ds = image_dataset_from_directory(
    "../animals", # Change according to the folder containing the dataset
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = image_dataset_from_directory(
    "../animals", # Change according to the folder containing the dataset
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(160,160,3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE+(3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights="imagenet")
image_batch,label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(3,activation="softmax")
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(160,160,3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs,outputs)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()

initial_epochs = 15
loss0,accuracy0 = model.evaluate(val_ds)
print("Initial loss: {:.2f} %".format(100*loss0))
print("Initial accuracy: {:.2f} %".format(100*accuracy0))
checkpoint = ModelCheckpoint("mobilenetv2.hdf5",monitor="val_accuracy",save_best_only=True,save_weights_only=False)
model.fit(train_ds,epochs=initial_epochs,validation_data=val_ds,callbacks=[checkpoint])
best = tf.keras.models.load_model("mobilenetv2.hdf5")
loss,accuracy = best.evaluate(test_ds)
print("\nTest acc: {:.2f} %".format(100*accuracy))
print("Test loss: {:.2f} %".format(100*loss))
