###############     Sequential Neural Network      ################
import tensorflow as tf
from tensorflow import keras

#   Helper libraries
import numpy as np
import matplotlib.pyplot as plt


#####   DATASET
#   MNIST Fashion Dataset (included in keras - pixel data of clothing images)   -   includes 60,000 images for training and 10,000 images for validation/testing
fashion_mnist = keras.datasets.fashion_mnist    # get dataset object

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    # load dataset and split into testing and training datasets

#   checkout / preview data
print("train_images.shape --> ")
print(train_images.shape)
print()
print("type(train_images) --> ")
print(type(train_images))
print()
print("checkout 1 pixes :: train_images[0,23,23] --> ")
print(train_images[0,23,23])
print()
    # pixels are values b/t 0-255, means grayscale... 0 = black, 255 = white
print("checkout first 10 training labels :: train_labels[:10] --> ")
print(train_labels[:10])
print()
    # labels are integers ranging 0 - 9. representing specific articles of clothing (classes)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# use matplotlib to actually view an image
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()



#####   DATA PREPROCESSING
train_images = train_images / 255.0     # squishing the images b/t 0-1.. for input values
test_images = test_images / 255.0


#####   BUILD THE MODEL
    #   Architecture Build      (define amt. neurons each layer, activation function, type of layer, type of connections)
model = keras.Sequential([      # Sequential is most basic neural network.. left -> right.. going through layer 'sequentially'.. inside here.. we will define the layers we want
    keras.layers.Flatten(input_shape=(28, 28)),     # input layer (1) -> flattening out our 28x28 pixel input to 1D pixels
    keras.layers.Dense(128, activation='relu'),     # hidden layer (2)  ->  Densely layered (all neurons in previous layer are connected to all neurons in this layer), 128 neurons (usually a good idea as a little smaller than input layer (sometimes bigger/ sometimes half the size.. / it depends)), activation function is REctified Linear Unit
    keras.layers.Dense(10, activation='softmax')    # output layer (3)  -> Densely layered, 10 output neurons, b/c 10 labels/classes to predict from, activation function is softmax will make sure sum output neuron values = 1 and each is b/t 0-1.
])
    #   Compile the Model       (defint optimizer, the loss, and the metrics we're looking at)  --> 'Hyper Parameter Tuning' -> things we can change to tune our model
model.compile(optimizer='adam',     #   algorithm that performs the gradient descent
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])     #   looking for accuracy from our model

#####   TRAIN THE MODEL
model.fit(train_images, train_labels, epochs=10)        #   pass the data, labels and epochs


#####   EVALUATE THE MODEL
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy: ', test_acc)

#####   HYPER PARAMETER TUNING
    # if test accuracy is less than model/training accuracy we over-fit our model (not as good at new data), can tune the parameters we can (have access to). Want the most generalized accuracy we can get.
        # Hyper parameter tuning - could change/tweak some of the architecture/optimizer/loss function/less(more) epochs/etc.
        # want highest accuracy possible on NEW data.. need to make sure our model generalizes properly


#####   MAKE PREDICTIONS
predictions = model.predict(test_images)    
predictions = model.predict([test_images[0]])   #   if doing only 1.. remember need to put it in an array, made for big data
print(np.argmax(predictions[0]))    # gives us the index of the max value in the list
print(class_names[np.argmax(predictions[0])])    # gives us class based on our class list we created






#####   VERIFY PREDICTIONS W/ VISUALS... don't need to necessarily run this
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
print("############# script complete ###############")

