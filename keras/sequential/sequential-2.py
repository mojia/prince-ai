from keras.models import Sequential
from keras import layers
from keras import optimizers


model = Sequential()
model.add(layers.Dense(name='fullConnection_2', units=3, activation='relu'))
model.add(layers.Dense(name='fullConnection_output',
                       units=1, activation='sigmoid'))


print(model.summary())
