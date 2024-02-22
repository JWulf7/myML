###############     Classification      ################
#to visualize... uncomment a visualization.... right click on code.. run in interactive window...run current file in interactive window

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pandas as pd

# this model uses features (sepal length, sepal width, petal length, and petal width) to classify flowers into labels - 3 different species (Setosa, Versicolor, Virginica)


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on


train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")


# saving the data into 2 seperate dateframes
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)       # (row 0 is the header)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe


# lets look at the data:
train.head()


#   Now we can pop the species column off and use that as our label.
train_y = train.pop('Species')
test_y = test.pop('Species')
train.head() # the species column is now gone

train.shape  # we have 120 entires with 4 features  (120, 4)


# define our input function
# different than Linear Regression, no epochs, and batch size different...
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


# create feature columns
# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():        # train.keys() , gives us all the columns.. could have also looped through CSV_COLUMN_NAMES and taken off 'Species'
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)       # output:   [NumericColumn(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='SepalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='PetalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='PetalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
# don't need to do the vocabulary and .unique from Linear Regression b/c already all encoded to numerics

# Instructions for updating:
#Use Keras preprocessing layers instead, either directly or via the `tf.keras.utils.FeatureSpace` utility. 
#Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.



# Build a DNN (Deep Neural Network classifier) with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.  (the 3 species we defined before...)
    n_classes=3)


# Train the model
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)     #   similar to an epoch... go through the data set until 5000 things have been looked at
# We include a lambda to avoid creating an inner function previously
# Since we need a function object, we create a function, that returns another function (the input function...)
# if look at the output of running this.... INFO:tensorflow:loss = 0.......  the loss, the lower the better


# now evaluate on test data
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



# make predictions
# create simple input function w/o the y value (or labels)
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

# allows user to input the features (petal lentgh, sepal length, etc...).. converts to a dataset for input into predict method
print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]   # need to put into a list.. b/c tensor flow works w/ multiple values better.. not really designed for making single input predictions

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    # prints the predicted class of the flower
    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))





# STEPS:
# 1. import necessary libraries and any other setup
# 2. load the data set(s)
# 3. explore the data set (make sure we understand it)
# 4. create input function(s)
# 5. create categorical and numeric columns (these will be in feature columns.... we don't have categorical in this proj.. therefore, combined w/ next step...)
# 5. create feature columns
# 7. create the model
# 8. train the model
# 9. evaluate model
# 10. make predictions




