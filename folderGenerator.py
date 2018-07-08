import os
import shutil
import json
import numpy as np
import sys

data_dir = 'C:/DataSets/Signs/augumentatedImg/'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
test_data_portion = 0.15
val_data_portion = 0.15


#функция для создания подкатологов
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    with open("annotations.json") as json_file:
        json_data = json.load(json_file)
        type = json_data["types"]
    for index in range(0, (len(type))):
        #print(type[index])
        os.makedirs(os.path.join(dir_name, type[index]))

#копирования изображений в данные каталоги
def copy_images(start_index, end_index, source_dir, dest_dir):
    files = os.listdir(data_dir)
    for index in range(start_index, end_index):#запись картинки в папку данного типа
         shutil.copy2(os.path.join(source_dir, files[index]), os.path.join(dest_dir, files[index][0:files[index].find(".")] ))

def countImagesOfTheType(directory):
    files = os.listdir(directory)  # получаем список файлов
    objects = []
    dct = {}
    with open("annotations.json") as json_file:
        json_data = json.load(json_file)
        type = json_data["types"]
    for index in range(0, len(files)):
        j = files[index].find(".")
        type = files[index][0:j]
        objects.append(type)
    for type in objects:
        if type in dct:
            dct[type] += 1
        else:
            dct[type] = 1
    return dct

# create_directory(train_dir)
# create_directory(val_dir)
# create_directory(test_dir)

typesCount = countImagesOfTheType(data_dir)

for index in range(0, len(typesCount)):
    files = os.listdir(data_dir)
    with open("annotations.json") as json_file:
        json_data = json.load(json_file)
        type = json_data["types"]
    try:
        nb_images = typesCount[type[index]]
    except KeyError:
        nb_images = 0
    start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
    start_test_data_idx = int(nb_images * (1 - test_data_portion))
    print(start_val_data_idx, start_test_data_idx, nb_images, type[index])
    files.index(0)
    #copy_images(0, start_val_data_idx, data_dir, train_dir)
    #copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
    #copy_images(start_test_data_idx, nb_images, data_dir, test_dir)
