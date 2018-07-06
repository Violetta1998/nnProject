import json
import re
import os

def getCharacteristics(num):
    with open("annotations.json") as json_file:
        json_data = json.load(json_file)
        fileName = str(json_data["imgs"][str(num)]["id"]) + '.jpg'
        fileOjects = str(json_data["imgs"][str(num)]["objects"])
        counts = fileOjects.count('category')
        characteristics = []
        for index in range(counts):
            if(json_data["imgs"][str(num)]["objects"]!=[]):
                characteristics.append(json_data["imgs"][str(num)]["objects"][index]["bbox"]["ymin"])
                characteristics.append(json_data["imgs"][str(num)]["objects"][index]["bbox"]["ymax"])
                characteristics.append(json_data["imgs"][str(num)]["objects"][index]["bbox"]["xmin"])
                characteristics.append(json_data["imgs"][str(num)]["objects"][index]["bbox"]["xmax"])
                characteristics.append(json_data["imgs"][str(num)]["objects"][index]["category"])
        res = fileName, characteristics
        with open('C:/DataSets/Signs/jsonObjects/'+str(num) + '.json', 'w') as outfile:
            json.dump(res, outfile)

def getImageNames(directory):
    files = os.listdir(directory) #получаем список файлов
    numbers = []
    for index in range(0, len(files)):
        numbers.append(files[index].replace(".jpg", ""))
    return numbers

testNumbers = getImageNames("C:/DataSets/Signs/test/")
trainNumbers = getImageNames("C:/DataSets/Signs/train/")
resultNumbers = trainNumbers
print(resultNumbers)

for index in range(0, len(resultNumbers)):
    getCharacteristics(resultNumbers[index])