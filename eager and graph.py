import tensorflow as tf
import time

x = tf.random.uniform((100,100))
y = tf.random.uniform((100,100))

# print(x)
def tensor_addition(x,y):
    return tf.reduce_sum(x+y)

start= time.time()
for i in range(1000):
    i = tensor_addition(x,y)
end= time.time()
print(f'Eager Execution time: {end-start:.4f}s')


@tf.function
def tensor_addition_graph(x,y):
    return tf.reduce_sum(x+y)

start= time.time()
for i in range(100):
    i = tensor_addition_graph(x,y)
end= time.time()
print(f'Graph Execution time: {end-start:.4f}s')