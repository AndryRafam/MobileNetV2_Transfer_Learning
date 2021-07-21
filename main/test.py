import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("mobilenetv2.hdf5")
names = ["Cat","Dog","Panda"] # must be in order
test = ["../test/1.jpg","../test/2.jpg","../test/3.jpg","../test/4.jpg","../test/5.jpg","../test/6.jpg"]

for name in test:
	img = tf.keras.preprocessing.image.load_img(name, target_size=(160,160))
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array,0)
	predictions = model.predict(img_array)
	score = predictions[0]
	plt.imshow(img)
	plt.axis("off")
	plt.show()
	print("{} ".format(names[np.argmax(score)]))
	print("\n")	


