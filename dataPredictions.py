import tensorflow as tf
from keras.models import Sequential, model_from_json
from teachNN import test_generator, train_generator
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

loaded_model = model_from_json(open("roadSigns.json", "r").read())
loaded_model.load_weights("roadSigns_weight.h5")

loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# test_images, test_labels = next(train_generator)
predictions = loaded_model.predict_generator(test_generator, steps=1, verbose=0)  # массив предсказаний
print(predictions)

# cm = confusion_matrix(test_labels, predictions[:, 0])


# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap = plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# cm_plot_labels = ['i4', 'i5', 'il60']
# plot_confusion_matrix(cm, cm_plot_labels, title='confusion matrix')
