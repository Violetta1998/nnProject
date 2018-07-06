import os
import shutil
import tensorflow as tf
import numpy as np
import sys

data_dir = 'C:\DataSets\Train'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
test_data_portion = 0.15
val_data_portion = 0.15
# Количество элементов данных в одном классе
nb_images = 12500

#функция для создания подкатологов
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "type1"))
    os.makedirs(os.path.join(dir_name, "type2"))#и так далее

#копирования изображений в данные каталоги -
# все изображения, которые принадлежат одному типу,
# получаем список из json, затем все эти номера пихаем в папку
def copy_images(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(source_dir, "type1." + str(i) + ".jpg"),
                    os.path.join(dest_dir, "type1"))
        shutil.copy2(os.path.join(source_dir, "type2." + str(i) + ".jpg"),
                   os.path.join(dest_dir, "type2"))

create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

#расчет индексов для обучения, проверки и тестиров
start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))

copy_images(0, start_val_data_idx, data_dir, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_images(start_test_data_idx, nb_images, data_dir, test_dir)