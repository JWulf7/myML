###############     Hidden Markov Model      ################

import tensorflow_probability as tfp
# install tensorflow_probability, run command in (git)bash terminal at project location: pip install tensorflow_probability
# didn't need to... but might need to add : --user
import tensorflow as tf

#   Goal    :   predict avg. temp of each day

#   weather model from: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel 
        # 1.    Cold days are encoded by a 0 and hot days are encoded by a 1.
        # 2.    The first day in our sequence has an 80% chance of being cold.
        # 3.    A cold day has a 30% chance of being followed by a hot day.
        # 4.    A hot day has a 20% chance of being followed by a cold day.
        # 5.    On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.

tfd = tfp.distributions # making a shorthand variable
# following based on weather model data from TensorFlow.org above
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])       #    probabilities of beginning in each state; day1: 80% cold, 20% warm
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],    #   probabilities for 1st state to stay in 1st state, and to transition to 2nd state; if cold today: 70% cold tomorrow, 30% warm tomorrow
                                                [0.2, 0.8]])    #   probabilities for 2nd state to transition to 1st state, and to stay in 2nd state; if warm today: 20% cold tomorrow, 80% warm tomorrow
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])   #   observation data; avg temp of [state 1, state 2], and standard deviation of [state 1, state 2]; temperatues *C in our case
#   loc argument represents the mean and the scale is the standard deviation

#   create model using distributions
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7 #   steps is how many days prediction is for (how many 'iteration cycles')
)
print("##################       model created successfully      ##################")

mean = model.mean() # takes probability from model... <-- 'partially defined tensor'

#   create/run session to run our tensor and get our result
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
    #   these are our temperatures it is predicting for the next 7 (steps) days