import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.list_physical_devices('GPU')
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

#for training data, random index value of data to be plotted range [0,59999]
data_idx = 19528
plt.figure()
plt.imshow(train_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid('False')
plt.show()

#display answer
train_labels[data_idx]

#for validation data, random index value of data to be plotted range [0,9999]
data_idx = 1957
plt.figure()
plt.imshow(train_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid('False')
plt.show()

# KEY = 0 T-shirt or top, 1 trousers, 2 pullover, 3 dress, 4 coat, 
# 5 sandala, 6 shirt, 7 sneaker, 8 bag, 9 ankle boots

# As each image is 28 x 28 pixels, and each pixel can have a value between 0 and 255 
# we will be assigning a weight to each pixel meaning we will have 784 weights 
# i.e. we will have 28 lists with 28 values each
valid_images[data_idx]

number_of_classes = train_labels.max() + 1
number_of_classes
model = tf.keras.Sequential
(
    [
        tf.keras.layers.Flatten(input_shape = (28,28)),
        tf.keras.layers.Dense(number_of_classes)
    ]
)

# checking if the model is as we expect it to be
model().summary()

# Create an instance of the Sequential class
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape = (28,28)),
        tf.keras.layers.Dense(number_of_classes)
    ]
)

# Now you can compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs = 10,
    verbose = 1,  # Use 1 for progress bar, 2 for one line per epoch
    validation_data=(valid_images, valid_labels)
)

#prediction time
model.predict(train_images[0:10])
data_idx = 42
plt.figure()
plt.imshow(train_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid('False')
plt.show()

x_values = range(number_of_classes)
plt.figure()
# Pass the image with the original shape (28, 28)
plt.bar(x_values, model.predict(train_images[data_idx:data_idx+1])[0])  
plt.xticks(range(10))
plt.show()
print("correct answers : ", train_labels[data_idx])