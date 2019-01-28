import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(4)

def generate_sequence(variable_length=False, pre_padding=True):
    if variable_length:
        x = []
        for _ in range(125000):
            x.append(np.random.randint(2, size=(np.random.random_integers(50))))
        if pre_padding:
            x = pad_sequences(x, maxlen=50, padding='pre')
        else:
            x = pad_sequences(x, maxlen=50, padding='post')
        x = np.asarray(x)
        x = x.reshape((125000, 50, 1))
        print(x.shape)
    else:
        x = np.random.randint(2, size=(125000, 50, 1))
    y = x.sum(axis=1) % 2
    return x, y

x, y = generate_sequence(variable_length=True, pre_padding=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

model = Sequential()
model.add(LSTM(32, input_shape=(50, 1)))
model.add(Dense(1, activation='sigmoid'))

callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
    )
]

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure(0)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure(1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

