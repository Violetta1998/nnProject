import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from teachNN import test_generator, train_generator, nb_test_samples, batch_size, test_dir
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

loaded_model = model_from_json(open("roadSigns.json", "r").read())
loaded_model.load_weights("roadSigns_weight.h5")

loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

test_img, test_labels = test_generator.next()
print(test_labels)

predictions = loaded_model.predict_generator(test_generator, steps = nb_test_samples // batch_size, verbose =0)
print(predictions)