import cv2
import random
import os
import json

defaultSize = 64
testDirectory = 'C:/DataSets/Signs/test/'
trainDirectory = 'C:/DataSets/Signs/train/'
jsonDirectory = 'C:/DataSets/Signs/jsonObjects/'
resultingDirectory = 'C:/DataSets/Signs/newData/'
errorDirectory = 'C:/DataSets/Signs/augumentatedImg/error/'

def getSign(name, xmin, xmax, ymin, ymax):
    img = cv2.imread(name)

    sign = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    sign = cv2.resize(sign, (0, 0), fx=defaultSize / (xmax - xmin),
                      fy=defaultSize / (ymax - ymin))  # приведение к размеру 64/64
    return sign

def cropImage(name, xmin, xmax, ymin, ymax):
    img = cv2.imread(name)
    newCropped = img[int(ymin) - 64:int(ymax) + 64, int(xmin) - 64:int(xmax) + 64]
    sign = getSign(name, xmin, xmax, ymin, ymax)

    result = newCropped.copy()
    x = y = 64
    result[y:y+sign.shape[0], x:x+sign.shape[1]] = sign

    #cv2.imshow("croppedImg", result)
    return result

def scaleImg(name, xmin, xmax, ymin, ymax):
    img = cv2.imread(name)

    newCropped = img[int(ymin) - 64:int(ymax) + 64, int(xmin) - 64:int(xmax) + 64]
    sign = getSign(name, xmin, xmax, ymin, ymax)

    height, width = sign.shape[:2]
    k = int(random.uniform(1, 3))
    sign = cv2.resize(sign, (k*width, k*height), interpolation=cv2.INTER_CUBIC)#изменение мастшаба рамки

    result = newCropped.copy()
    if k == 1:
        x = y = 60
    else:
        x = y = 35
    result[y:y + sign.shape[0], x:x + sign.shape[1]] = sign

    #cv2.imshow("scaledImg", result)
    return result

def translateImg(name, xmin, xmax, ymin, ymax):
    img = cv2.imread(name)
    newCropped = img[int(ymin) - 64:int(ymax) + 64, int(xmin) - 64:int(xmax) + 64]

    sign = getSign(name, xmin, xmax, ymin, ymax)

    result = newCropped.copy()
    x = y = int(random.uniform(30, 100))
    result[y:y + sign.shape[0], x:x + sign.shape[1]] = sign

    #cv2.imshow("translatedImg", result)
    return result

def rotateImg(name, xmin, xmax, ymin, ymax):
    img = cv2.imread(name)
    newCropped = img[int(ymin) - 64:int(ymax) + 64, int(xmin) - 64:int(xmax) + 64]

    sign = getSign(name, xmin, xmax, ymin, ymax)

    rows, cols = sign.shape[:2]
    angle = random.choice([-90, 90, -180, 180])
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(sign, M, (cols, rows))

    result = newCropped.copy()
    x = y = 70
    result[y:y + sign.shape[0], x:x + sign.shape[1]] = dst

    #cv2.imshow("rotatedImg", result)
    return result

def resultingImage(img, name, dir):
    img = cv2.resize(img, (0, 0), fx=32 / (img.shape[0]), fy=32 /img.shape[1])
    cv2.imwrite(dir + name, img)
    return img

def createDataset(jsonObject):
    with open(jsonDirectory + jsonObject) as json_file:
        json_data = json.load(json_file)
        coordinates = []
        signsCount = len(json_data[1]) / 5  # кол-во знаков на изображении
        for index in range(0, (int(len(json_data[1]))), 5):
            y1 = json_data[1][index]
            y2 = json_data[1][index + 1]
            x1 = json_data[1][index + 2]
            x2 = json_data[1][index + 3]
            type = json_data[1][index + 4]
            res = x1, x2, y1, y2, type
            coordinates.append(res)
        img = []
        imgtest = []

        imgPath = trainDirectory + json_data[0]

        try:
            imgtest = cropImage(imgPath, coordinates[0][0], coordinates[0][1], coordinates[0][2], coordinates[0][3])
        except TypeError:
            imgPath = testDirectory + json_data[0]
        except ValueError:
            try:
                img = cv2.imread(imgPath)
                resultingImage(img, json_data[0], errorDirectory)
            except IndexError:
                img = cv2.imread(imgPath)

        for i in range(0, int(signsCount)):
            rand = random.choice([0, 1, 2, 3])
            try:
            # if(coordinates[i][4] == "io"):
                if rand == 0:
                    img = scaleImg(imgPath, coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3])
                    resultingImage(img, str(coordinates[i][4]) + "." + str(i) + "." + json_data[0], resultingDirectory)
                if rand == 1:
                    img = translateImg(imgPath, coordinates[i][0], coordinates[i][1], coordinates[i][2],
                                       coordinates[i][3])
                    resultingImage(img, str(coordinates[i][4]) + "." + str(i) + "." + json_data[0], resultingDirectory)
                if rand == 2:
                    img = rotateImg(imgPath, coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3])
                    resultingImage(img, str(coordinates[i][4]) + "." + str(i) + "." + json_data[0], resultingDirectory)
                if rand == 3:
                    img = cropImage(imgPath, coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3])
                    resultingImage(img, str(coordinates[i][4]) + "." + str(i) + "." + json_data[0], resultingDirectory)
                print("success")
            except ValueError:
                img = cv2.imread(imgPath)
                resultingImage(img, json_data[0], errorDirectory)
            except IndexError:
                img = cv2.imread(imgPath)
                resultingImage(img, json_data[0], errorDirectory)

def getJsonName(directory):
    files = os.listdir(directory)#получаем список файлов
    objects = []
    for index in range(0, len(files)):
        objects.append(files[index])
    return objects

jsonObj = getJsonName(jsonDirectory)
for index in range(8100, len(jsonObj)):
    createDataset(jsonObj[index])

cv2.waitKey(10000000)