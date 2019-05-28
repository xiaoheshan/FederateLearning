import os
import threading
import json
from simple_layer import *


# runs in thread
def runProgram(threadID, iteration):
    os.system("python mnist.py " + str(threadID) + " " + str(iteration))


def load_json(name):
    with open(name, "r") as load_json:
        return json.load(load_json)


def listToNumpy(parameters):
    W1 = np.array(parameters["W1"])
    b1 = np.array(parameters["b1"])
    W2 = np.array(parameters["W2"])
    b2 = np.array(parameters["b2"])
    W3 = np.array(parameters["W3"])
    b3 = np.array(parameters["b3"])
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }
    return parameters


def getAveModel(models):
    lenOfModels = len(models)
    # get zero matrix
    resW1 = models[0]['W1'] - models[0]['W1']
    resb1 = models[0]['b1'] - models[0]['b1']
    resW2 = models[0]['W2'] - models[0]['W2']
    resb2 = models[0]['b2'] - models[0]['b2']
    resW3 = models[0]['W3'] - models[0]['W3']
    resb3 = models[0]['b3'] - models[0]['b3']

    for model in models:
        resW1 = resW1 + model['W1'] / lenOfModels
        resb1 = resb1 + model['b1'] / lenOfModels
        resW2 = resW2 + model['W2'] / lenOfModels
        resb2 = resb2 + model['b2'] / lenOfModels
        resW3 = resW3 + model['W3'] / lenOfModels
        resb3 = resb3 + model['b3'] / lenOfModels

    parameters = {
        'W1': resW1.tolist(),
        'b1': resb1.tolist(),
        'W2': resW2.tolist(),
        'b2': resb2.tolist(),
        'W3': resW3.tolist(),
        'b3': resb3.tolist()
    }
    return parameters


def save_model(name, parameter):
    with open(name, 'w') as json_file:
        json.dump(parameter, json_file, ensure_ascii=False)


if __name__ == '__main__':

    models = []
    for i in range(10):
        model = listToNumpy(load_json("./singleTrainModel/model{}.json".format(i)))
        models.append(model)
    aveModel = getAveModel(models)
    resPath = "./polymerizeModel/modelPolymerizationResult.json"
    save_model(resPath, aveModel)
