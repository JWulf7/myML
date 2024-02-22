#####       LINEAR REGRESSION       ########

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from six import urllib
import urllib

# import tensorflow.compat.v2.feature_column as fc
from tensorflow import feature_column as fc

import tensorflow as tf


print()
print()
print()
print()
print("::::::::::::::::LOADING PY SCRIPT HERE::::::::::::::::")


# Load dataset.
# loads datasets into a pandas dataframe object
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data - to train the data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data - to test the data


print("head (first 5 entries) of dftrain ==>")
print(dftrain.head())
# popping off 'survived column from df objects, saved those columns to new y_... objects
# this will be our label
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print("head of dftrain ==> ")
print(dftrain.head())
print()
print()
print()
print("head of y_train ==>")
print(y_train.head())
print()
print()
print()
print("shape of dftrain ==> ")
print(dftrain.shape)
print()
print()
print()
#gives us a histogram of the age
print("histogram of age ==>")

#to visualize... uncomment a visualization.... right click on code.. run in interactive window...run current file in interactive window
# just visualizing a few things to get some ideas of what we're working with
# dftrain.age.hist(bins=20)
print()
print()
print()
print("plot of sex ==>")
# dftrain.sex.value_counts().plot(kind='barh')
print()
print()
print()
print("plot of class ==>")
# dftrain['class'].value_counts().plot(kind='barh')
print()
print()
print()
print("plot of survival by sex ==> ")
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
print()
print()
print()




CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']   # non-numerical data values
# always need to transform categorical values into numbers some how.  we encode the data using integer values
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
# loops through categorical columns array, for each entry/'feature_name', we define a vocabulary which is 
#   equal to the dataframe at that feature name and get all the different unique values
#   then... appends to feature_columns, creates a column that has the feature_name, and then all the different vocabulary associated with it
# basically turning categorical data to numerical data....
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

# loops through numeric columns array, for each entry/'feature name', we append to feature_columns array a new column
#   can give the feature_name, and datatype, and create a column with that...
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print("print(feature_columns) ==> ")
print(feature_columns)
print()
print()
print()


#Create input function to define how our data is going to be broken into epochs and into batches to feed to our model

# takes our data, and encodes it in a tf.data.Dataset object... b/c our model needs a tf.data.Dataset object to create the model
# need to take the pandas dataframe, and turn it into a tf.data.Dataset object....
#       creates an input function, returns that function
#       params: data_df : our pandas dataframe; label_df : our labels (the y_train or y_eval) ; num_epochs : number of epochs to do.. we set default to 10 ; shuffle : are we going to shuffle/mixup our data  ; batch_size : how many data points to give the model while training at once ;
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    # essentially pass a dictionary representation of our dataframe as keys and label dataframe as values -> create the tf.data.Dataset object from that
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)  # not shuffling here + only doing 1 epoch.. b/c not training.. only evaluating...


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier to the LinearClissifier object in TensorFlows estimator module


linear_est.train(train_input_fn)  # train by using estimators train method, give it the input function...
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model
print(result)

#can mess w/ number of epochs to make more or less accurate....


# now to use the model to make predictions.... 

result = list(linear_est.predict(eval_input_fn))
## converting to a list so that we can loop through it.
print(result)
print(result[0])
print(dfeval.loc[0])    # prints the first persons info from eval list
print(y_eval.loc[0]) # whether they survived or not... 1 = yes, 0 = dead
print(result[0]['probabilities'][1])    # chance of survival of first person from eval list...













#STEPS:
# 1. import necessary libraries
# 2. load the data set(s)
# 3. explor the data set (make sure we understand it)
# 4. create categorical and numeric columns
# 5. for a linear estimator, need to create these as feature columns
# 6. create input function(s)
# 7. create the model
# 8. train the model
# 9. evaluate model
# 10. make predictions
