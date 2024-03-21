###############     Recurrent Neural Network      ################
###############     Text Generation Sequential Model      ################

# this will generate text to create a play based on training data from Romeo and Juliet
# https://www.tensorflow.org/tutorials/text/text_generation

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

############        LOAD DATASET        ############
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')        #saving file from location as shakespeare.txt and assigning to variable

###### can also load our own text file:
# from google.colab import files
# path_to_file = list(files.upload().keys())[0]

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

# take a look at the first 250 characters in the text
print(text[:250])


############        DATA PREPROCESSING/ENCODING        ############
# encode each unique character as a different integer
vocab = sorted(set(text))   # figure how many unique characters
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}       # creates mapping of each unique character to an index as it iteraters over vocab
idx2char = np.array(vocab)      # creating array of vocab so we can use index a character appears as a reverse mapping

# simple function to turn each character into integer representation of input text and return as an array
def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)     # input dataset text, now as a representative integer array


# peek encoding
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

# reverse int to text function
def int_to_text(ints):
    try:
        ints = ints.numpy() # make sure input is numpy array
    except:
        pass
    return ''.join(idx2char[ints])
# check
print(int_to_text(text_as_int[:13]))


############        CREATE TRAINING EXAMPLES        ############
# need to split our text data from above into many shorter sequences that we can pass to model as training examples
# training examples will use a seq_length sequence as input and a seq_length sequence as output where the sequence is the original sequence shifted one letter to the right
# ex:       input: Hell     |       output: ello 

# Step 1: create a stream of characters from our text data
seq_length = 100    # length of sequence for a training example
exmples_per_epoch  = len(text)//(seq_length+1)      # text / 100 seq length... but need 101 chars per training example... first 100 for input.. last 100 for output..   i.e. Hell -> ello

# Step 2: create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)  # converts our entire string dataset into characters

# Step 3: use batch method to turn stream of characters into batches of desired length... drop any remaining chars that don't fit into a batch
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Step 4: use the sequences of 101 chars and split into input and outputs
def split_input_target(chunk):      # ex:   Hello
    input_text = chunk[:-1]         #       hell
    target_text = chunk[1:]         #       ello
    return input_text, target_text  #       Hell, ello

dataset = sequences.map(split_input_target)     # we use map to apply the above function to every entry

# peek
for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_text(x))
    print("\nOUTPUT")
    print(int_to_text(y))

# Step 5: make training batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)     # vocab is number of unique characters
EMBEDDING_DIM = 256     # embedding dimension   how big the vectors that represent our words are
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# shuffle dataset and put them into batches
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


############        CREATE MODEL        ############
# going to make function that will return a model, b/c going to train model w/ a batch size of 64; later going to save model parameters except batch size, and pass it batches of 1, so it can make predictions 1 char at a time
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),    # None = don't know how long sequences will be in each batch.... currently to train 101, but when utilizing model for predictions, don't know length of sequences
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,     # return the intermediate stage at every step
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),   # picking this b/c TF says good default to pick
        tf.keras.layers.Dense(vocab_size)       # we want final layer to have potential output of all unique characters
    ])
    return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

############        CREATE CUSTOM LOSS FUNCTION        ############
# creating custom loss function b/c model will output a (64, sequence_length(100), 65) (batch size, sequence length, vocab size) shaped tensor that represents probability distribution of each character at each timestep for every sequence in the batch.
# but will be changing batch size later

# lets peek sample input and output to see what model actually gives us w/o training
for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)      # ask our model for a prediction on our first batch of training data (64 entries)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")       # print out the output shape

# we can see that the prediction is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)

# lets examine one prediction
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each step

# finally we'll look at a prediction at the first timestep
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course it's 65 values representing the probability of each character occuring next

# if we want to determine the predicted character we need to sample the output distribution (pick a value based on probability)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars     # this is what the model predicted for training sequence 1

# now need to create loss function that can compare that output to the expected output and give some numeric value representing how close the two were
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


############        COMPILE THE MODEL        ############
model.compile(optimizer='adam', loss=loss)


############        CREATE CHECKPOINTS        ############
# going to setup and configure model to save checkpoints as it trains. Will allow us to load our model from a checkpoint and continue training it.

# Directory where the checkpoint will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


############        TRAIN THE MODEL        ############
history = model.fit(data, epochs=40, callbacks=[checkpoint_callback])


############        LOAD THE MODEL        ############
# we'll rebuild the model from a checkpoint using a batch_size of 1, so that we can feed 1 piece of text to model and have it make prediction
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# we'll use the last checkpoint that stores the models weights 
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# could also load ANY checkpoint by specifying exact file to load
# checkpoint_num = 10
# model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
# model.build(tf.TensorShape([1, None]))


############        TEXT GENERATION        ############
# can use function provided by TensorFlow to generate some text using any starting string we'd like
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 800

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting. 
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))

