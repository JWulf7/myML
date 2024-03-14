#Problem    classifying 10 different everyday objects
#Dataset    CIFAR Image Dataset (already pre-loaded into keras)- containse 60,000 32x32 color images w/ 6,000 images of each class
#Labels ->  Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator



#####   LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()     # this time it loads our dataset as a TensorFlow Dataset object; Not in like a numpy array

#Normalize pixel values to be b/t 0 - 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

#####   Peek an image
IMG_INDEX = 1   # change this to look at other images

plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()



##########      CNN ARCHITECTURE    ###########
### Build the Convolutional Base    - extracts the features out of your image(s)    -   will be input for our dense layers
model = models.Sequential()     #stacking a bunch of convolutional layers and (max/min/avg) pooling layers together
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))    # Conv2D(32 = amount of filters), (3, 3)= sample size (... how big are the filters).. activation function 'rectified linear unit' applied after dot-product.. input_shape(define the input shape of this layer(in this case our input images... need this in out first layer))
model.add(layers.MaxPooling2D((2, 2)))  #   2, 2    = 2x2 sample size w/ a stride of 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))     # don't need to define input_shape b/c they will figure it out based on input from previous layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# let's peek the model
model.summary()

### Adding Dense Layers -   Classifier
model.add(layers.Flatten())     #   take our output from the base, and flatten into 1 dimension
model.add(layers.Dense(64, activation='relu'))  # 64 neuron dense layer, w/ rectifier linear unit activation function
model.add(layers.Dense(10)) #   output layer, 10 neuron dense layer, 10 b/c # of classes we have for classification

model.summary()


##########      TRAINING    ###########
model.compile(optimizer='adam',     # can see nN1Sequential.py for more details
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # loss function - CategoricalCrossentropy good basic loss function... SparseCategoricalCrossentropy good for classification tasks
                metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


##########      EVALUATING THE MODEL    ###########
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)



##########      DATA AUGMENTATION    ###########
#   when working with small datasets (1,xxx's instead of 1,xxx,xxx's we can augment our data to avoid overfitting )


# creates a data generator object that transforms images
datagen = ImageDataGenerator (
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# pick an image to transform
test_img = train_images[14]
img = image.img_to_array(test_img)  # convert image to numpy array
img = img.reshape((1,) + img.shape) # reshape image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):     # loop runs forever until we break, saving images to current directory
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:   # show 4 images
        break

plt.show()

print("############ DONE WITH SCRIPT EXECUTION  ############")

