import math
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
from keras.optimizers import SGD

train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
img_width, img_height = 32, 32
# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_width, img_height, 3)
epochs = 30
batch_size = 16

nb_train_samples = 1135
nb_validation_samples = 240
nb_test_samples = 240


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Слои с 1 по 6 используются для выделения важных признаков в изображении, а слои с 7 по 10 - для классификации.
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# компилируем нейронную сеть
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# learning schedule callback
lrate = LearningRateScheduler(step_decay)
tbCallBack = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [lrate, tbCallBack]

sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1] * 100))

model_json = model.to_json()
json_file = open("roadSigns.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("roadSigns_weight.h5")

