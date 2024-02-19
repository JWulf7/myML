import tensorflow as tf
print()
print()
print(tf.version)
print("Hello World")


rank1_tensor = tf.Variable(["word", "another word"], tf.string)
rank2_tensor = tf.Variable([["word", "yes"],["another word", "no"]], tf.string)
print('rank1_tensor = ')
print(tf.rank(rank1_tensor))
print('rank2_tensor = ')
print(tf.rank(rank2_tensor))


