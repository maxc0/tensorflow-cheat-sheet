# TENSORFLOW CHEAT SHEET

# IMPORTANT TENSORFLOW
import tensorflow as tf

# PRINT VERSION
print(tf.__version__)

# CREATE TENSORS
constant_scalar = tf.constant(7)
constant_vector = tf.constant([1,2])
constant_matrix = tf.constant([ [1,2], [3,4] ])

variable_scalar = tf.Variable(7)
variable_vector = tf.Variable([1,2])
variable_matrix = tf.Variable([ [1,2], [3,4] ])

# PRINT RANK, SHAPE, SIZE, DATA TYPE
print(constant_scalar.ndim)
print(constant_scalar.shape)
print(tf.size(constant_scalar))
print(constant_scalar.dtype)

# CREATE TENSORS WITH SPECIFIC DATA TYPE
float_tensor = tf.constant([ [1,2], [3,4] ], dtype=tf.float16)

# CREATE RANDOM TENSORS
random_tensor_generator = tf.random.Generator.from_seed(42)
normal_distribution_tensor = random_tensor_generator.normal(shape=(4,2),  dtype=tf.float16)
uniform_distribution_tensor = random_tensor_generator.uniform(shape=(4,2), dtype=tf.float16)

# SHUFFLE TENSOR (ONLY AT THE HIGHEST DIMENSION)
t1.normal(shape=(10000, 50,50),  dtype=tf.float16) # e.g. 10,000 50x50 images
t1 = tf.random.shuffle(t1) # the 10,000 images are now shuffled

# CREATE TENSORS OF ZEROES AND ONES
t2 = tf.ones(shape=(10,2))
t3 = tf.zeros(shape=(10,2))

# ELEMENT-WISE TENSOR OPERATIONS
# Refer to tf.math, https://www.tensorflow.org/api_docs/python/tf/math, for more details
tf.add(t1, 10)
tf.multiply(t1, 10)
tf.abs(t1)
tf.reduce_min(t1) # minimum | returns scalar
tf.reduce_max(t1) # maximum | returns scalar
tf.reduce_mean(t1) # mean | returns scalar
tf.reduce_sum(t1) # sum | returns scalar
tf.reduce_variance(t1) # variance | returns scalar
tf.reduce_std(t1) # standard deviation | returns scalar

# MATRIX MULTIPLICATION
# inner dimensions must match
# resultant matrix is the same shape as the outer dimensions
t4 = tf.constant([[1,2,5], [7,2,1], [3,3,3]]) # 3x3
t5 = tf.constant([[3,5], [6,7], [1,8]]) # 3x2
tf.matmul(t4, t5) # 3x3

# TRANSPOSE
tf.transpose(t1)

# RESHAPE
# the size of both tensors must be the same
tf.reshape(t3, [2,3])

# CHANGE DATA TYPE
# tf.int32 is default
t1 = tf.cast(t1, dtype=tf.float16)

# POSITIONAL MAX AND MIN
# returns a tensor containing the index of the largest value in each axis
tf.argmax(t1)

# SQUEEZING TENSORS
# remove all dimensions of size 1
tf.squeeze(t1)

