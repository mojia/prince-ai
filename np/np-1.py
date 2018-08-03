import numpy as np

a = np.array([[1, 2], [3, 4]])

sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)
print(sum0)
print(sum1)

print('------')

data = 10 * np.random.random((6, 6))
print(data)
print('data min ' + str(data.min()))

print('------')


# a = np.array([[1, 2, 4], [3, 4, 1]])
a = np.array([[5, 6, 3, 1], [1, 7, 8, 9], [2, 4, 8, 2]])
b = np.array([[1], [2], [3]])

x = np.hstack((a, b))
print(str(x))

print('------')

train_x = np.random.random(10)
for i in range(len(train_x)):
    print(str(train_x[i]))


print('-------------------------------')
a = 10 * np.random.random((3, 20))
print(a)
print(a[:, 3:6])
print(a[:, 6:9])


print('-------------------------------')
index = 0
step = 3
input_train = [5, 1, 23, 2, 1, 11, 78, 9, 4, 6, 2, 1]
input_kLines_train = []
while (index < len(input_train)):
    input_kLines_train.append(input_train[index])
    index += step

print(input_kLines_train)
