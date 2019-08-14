from sklearn.model_selection import StratifiedKFold
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling1D,Reshape,Flatten,InputLayer
from keras.layers import Conv1D, MaxPooling1D,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils


Fmnist_train, Fmnist_test= tf.keras.datasets.fashion_mnist.load_data()
Fmnist_train_X, Fmnist_train_Y = Fmnist_train
Fmnist_test_X, Fmnist_test_Y = Fmnist_test

def shaping(data, target):
    data = np.array(data, dtype=np.uint8)
    target = np.array(target, dtype=np.uint8)
    data = data.reshape(data.shape[0], 28, 28, 1)
    target = np_utils.to_categorical(target, 10)
    data = data.astype('float32')
    data /= 255
    return data, target

X_train_shaped, Y_train_shaped = shaping(Fmnist_train_X, Fmnist_train_Y)
X_test_shaped, Y_test_shaped = shaping(Fmnist_test_X, Fmnist_test_Y)
model = Sequential()
model.add(Conv2D(64,(4,4),activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(64,(4,4),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
#model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_shaped,Y_train_shaped, validation_split = 0.33,epochs = 3,verbose = 1)
#_,accuracy = model.evaluate(Fmnist_test_X, Fmnist_test_Y, batch_size = 15, verbose=1)
#print(accuracy)

