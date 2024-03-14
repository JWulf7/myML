from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

#   script for data augmentation... can replace loaded dataset... might not need to convert to numpy array(depending on source)

#####   LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()     # this time it loads our dataset as a TensorFlow Dataset object; Not in like a numpy array

#Normalize pixel values to be b/t 0 - 1
train_images, test_images = train_images / 255.0, test_images / 255.0



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
img = image.img_to_array(test_img)  # convert image to numpy array from the TensorFlow Dataset object
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