###############     Recurrent Neural Network      ################
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np
import keras



############        LOAD DATASET        ############
# IMDB movie review dataset from keras. 
# This dataset contains 25,000 reviews from IMDB where each one is already preprocessed and has a label as either positive or negative. 
# Each review is encoded by integers that represents how common a word is in the entire dataset. For example, a word encoded by the integer 3 means that it is the 3rd most common word in the dataset.

VOCAB_SIZE = 88584      # unique words

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

# peek our data
train_data[0]
len(train_data[0])
len(train_data[1])

############        DATA PREPROCESSING        ############
# reviews are different lengths...
# need to pad our sequences
    # if the review is greater than 250 words then trim off the extra words
    # if the review is less than 250 words add the necessary amount of 0's to make it equal to 250.
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

############        CREATE MODEL        ############
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),      # embedding layer.. even though dataset has preprocessed inputs, embedding layer will find more meaningful representations w/ vectors than initial integer values; 32= output dimensions of all vectors
    tf.keras.layers.LSTM(32),                       # LSTM layer 32 dimensions for every single word
    tf.keras.layers.Dense(1, activation="sigmoid")  # Dense layer, 1 Node output, sigmoid(squishes values b/t 0-1)... b/c classifying sentiment... 0-0.5 negative review, 0.5-1 positive review
])
model.summary()

############        TRAINING        ############
# compile and train the model
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])     # loss function = binary_crossentropy b/c 2 different things we could be predicting. optimizer = rmsprop, not crazy important.. could use adam if you wanted to 

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)      # validation_split = 0.2 b/c using 20% of training data to evaluate model as we go through
# evaluation accuracy stalls ~ 88%... whereas model gets overfit at ~97-98%     --> what this tells us is we don't have enough training data
# we'll leave it for now
results = model.evaluate(test_data, test_labels)
print(results)
# accuracy of ~ 85-86%... not great but not bad for a simple recurrent NN

############        MAKING PREDICTIONS        ############
# Since our reviews are encoded well need to convert any review that we write into that form so the network can understand it. 
# To do that well load the encodings from the dataset and use them to encode our own data.

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens] # if word exists in word_index we get from imdb mapping, use mapping; else use 0
    return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

# can also make a decode function
reverse_word_index = {value : key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "

    return text[:-1]

print (decode_integers(encoded))
# it works

# now time to make prediction
def predict(text):
    encoded_text = encode_text(text)    # encode our review... then insert into a numpy array of length 250.. return result
    pred = np.zeros((1,250))        # 250 = shape our model expects for input (length of movie review)
    pred[0] = encoded_text              # need to put into an array b/c model is optimized for multiple entries... so it requires input array
    result = model.predict(pred)
    print(result[0])


positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great"
predict(positive_review)

negative_review = "That movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)

positive_review2 = "loved it. That movie was banging. Tom Cruise is the best actor! Highly recommend"
predict(positive_review2)

positive_review3 = "The movie was really really good"
predict(positive_review3)

predict(text)