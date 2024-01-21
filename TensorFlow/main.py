import tensorflow as tf
import numpy as np

#Вектор
vector = tf.constant([5, 5])

print("shape:", vector.shape)
print("rank:", tf.rank(vector))
print("size:", tf.size(vector).numpy())

#Скаляр
scalar = tf.constant(7)

print("shape:", scalar.shape)
print("rank:", tf.rank(scalar))
print("size:", tf.size(scalar).numpy())

#Матрица
matrix = tf.constant([[1, 5, 3], [4, 2, 6]])

print("shape:", matrix.shape)
print("rank:", tf.rank(matrix))
print("size:", tf.size(matrix).numpy())

#Тензор
tensor = tf.constant([[[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 0]]])

print("shape:", tensor.shape)
print("rank:", tf.rank(tensor))
print("size:", tf.size(tensor).numpy())

# Создание двух тензоров со случайными значениями от 0 до 1, размерностью (5, 300)
random_tensor_1 = tf.constant(np.random.rand(5, 300))
random_tensor_2 = tf.constant(np.random.rand(5, 300))

print(random_tensor_1)
print(random_tensor_2)

#Умножение с использованием матричного умножения
result_matrix_multiply = tf.matmul(random_tensor_1, tf.transpose(random_tensor_2))

print(result_matrix_multiply)

#Умножение с использованием скалярного произведения
result_dot = tf.tensordot(random_tensor_1, tf.transpose(random_tensor_2), axes=1)

print(result_dot)

#Создание тензора со случайными значениями в области значений от 0 до 1 размерностью [224, 224, 3]
random_tensor = tf.constant(np.random.rand(224, 224, 3))
print(random_tensor)

#Нахождение минимального и максимального значений
min_value = tf.reduce_min(random_tensor)
max_value = tf.reduce_max(random_tensor)

print("Min:", min_value.numpy())
print("Max:", max_value.numpy())

#Создание тензора с случайными значениями формы [1, 224, 224, 3]
random_tensor = tf.random.normal(shape=(1, 224, 224, 3))

#Изменение формы на [224, 224, 3] с использованием squeeze
squeezed_tensor = tf.squeeze(random_tensor)

print("Initial Form:", random_tensor.shape)
print("Modified Form:", squeezed_tensor.shape)

# Создание тензора размером 10
dec_tensor = tf.constant([5, 8, 23, 6, 5, 4, 8, 9, 5, 3])

# Поиск индекса максимального значения
max_index = tf.argmax(dec_tensor).numpy()

print("Original Tensor:", dec_tensor.numpy())
print("Index with Maximum Value:", max_index)

dec_tensor = tf.one_hot(dec_tensor, depth=10)

print("One-hot Encoding Tensor: ", dec_tensor)