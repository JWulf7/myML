#####       Using a Pretrained model and fine tuning it     #####
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras

tfds.disable_progress_bar()

############        LOAD DATASET        ############
# needed to add a line to a fix naming bug here to fix below : C:\Users\Owner\AppData\Roaming\Python\Python311\site-packages\tensorflow_datasets\image_classification\cats_vs_dogs.py ln 117-119
#   split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

############        DATA PREPROCESSING        ############

IMG_SIZE = 160  # all images will be resized to 160x160 ... currently they are not all the same size

def format_example(image, label):
    """
    returns an image that is reshaped to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)  # convert every pixel in image to a float b/c could be int's
    image = (image/127.5) - 1               # converting every 255 color value b/t 0 - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))    # resize input image to 160 x 160
    return image, label

# apply resize formatting to all/each our images in each of the split datasets
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# lets peek
for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# shuffle and batch the images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# shape of orig image vs new image
for img, label in raw_train.take(2):
    print("Original shape:", img.shape)

for img, label in train.take(2):
    print("New shape:", img.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

############        CREATE PRETRAINED CONVOLUTIONAL BASE        ############
#create the base model from the pre-trained model MobileNet V2  -> developed at Google and trained on 1.4 Million images and has 1,000 different classes
base_model = tf.keras.applications.MobileNetV2(input_shape= IMG_SHAPE,      # tf.keras.applications.MobileNetV2 tells us the architecture... define input shape
                                               include_top=False,       # exclude the classifier .. no.. only the convolutional base
                                               weights='imagenet')      # a specific save of the weights
base_model.summary()
# base_model will output a shape of (32, 5, 5, 1280) -> 5, 5, 1280 is output of our base_model "out_relu" ... 32 layers of different filters/features
for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
print(feature_batch.shape)


############        FREEZE THE BASE        ############
# obviously we don't want to retrain the base
base_model.trainable = False
base_model.summary()    # now 0 trainable params


############        ADD OUR CLASSIFIER        ############
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()     # flattening into 1D tensor by taking entire avg. of 1280 layers that are 5x5

prediction_layer = keras.layers.Dense(1)    # out put layer.. 1 node.. b/c only 2 classes

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.summary()


############        TRAINING THE MODEL        ############
base_learning_rate = 0.0001     # picked a slow/low learning rate, b/c already have a base model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),    # loss function = BinaryCrossentropy b/c using 2 classes as output ;    more than 2 classes some type of CrossEntropy
              metrics=['accuracy'])

# we can evaluate the model to see how it does before training it on our new images
initial_epochs = 3
validation_steps = 20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
# before training, accuracy ~ 51.88%

# train it on images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

# save model by...
# model.save("dogs_vs_cats.h5")       # .h5 keras specific model format ... seems deprecated.. changing to .keras
model.save("dogs_vs_cats.keras")

# can load model by...
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')   # obviously change file format to .keras if did that...

# use model
#model.predict(some, inputs, here)


