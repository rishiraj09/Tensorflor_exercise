import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

const_node_1 = tf.constant(1.0, dtype=tf.float32)
const_node_2 = tf.constant(2.0, dtype=tf.float32)
const_node_3 = tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)

adder_node_1 = tf.add(const_node_1, const_node_2)
adder_node_2 = const_node_1 + const_node_2
mult_node_1 = adder_node_2 * const_node_3

session = tf.Session()

print(session.run(mult_node_1))

placeholder_1 = tf.placeholder(dtype=tf.float32)
placeholder_2 = tf.placeholder(dtype=tf.float32)


multiply_node_1 = placeholder_1 * 3
multiply_node_2 = placeholder_1 * placeholder_2

session = tf.Session()
print(session.run(multiply_node_2,{placeholder_1: 4.0, placeholder_2: [1.0,2.0,3.0]}))

var_node_1 = tf.Variable([5.0], dtype=tf.float32)
const_node1 = tf.constant([10.0], dtype=tf.float32)


session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
print(session.run(var_node_1))