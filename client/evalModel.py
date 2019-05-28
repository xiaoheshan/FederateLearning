from neural_network import *
from simple_layer import *
import json

test_images = np.load('./data/testImages.npy')
test_labels = np.load('./data/testLabels.npy')

def load_json(name):
    with open(name, "r") as load_json:
        return json.load(load_json)

def save_txt(path, value):
    value = value + " "
    with open(path, 'a') as f:
        f.write(value)

# for i in range(11):
#     primModel = load_json("./polymerizeModel/resModel{}.json".format(i))
#
#     model = [
#     LinearLayer(primModel['W1'], primModel['b1']),
#     ReluLayer(),
#     LinearLayer(primModel['W2'], primModel['b2']),
#     ReluLayer(),
#     LinearLayer(primModel['W3'], primModel['b3']),
#     SoftmaxOutputLayer()
#     ]
#     test_accuracy_value = calculate_accuracy(model, test_images, test_labels)
#     accPath = "./accuracy/polymerizeAccuracy.txt"
#     save_txt(accPath,str(test_accuracy_value))
#     print(test_accuracy_value)


primModel = load_json("./polymerizeModel/modelPolymerizationResult.json")

model = [
LinearLayer(primModel['W1'], primModel['b1']),
ReluLayer(),
LinearLayer(primModel['W2'], primModel['b2']),
ReluLayer(),
LinearLayer(primModel['W3'], primModel['b3']),
SoftmaxOutputLayer()
]

test_accuracy_value = calculate_accuracy(model, test_images, test_labels)
accPath = "./accuracy/modelPolymerizeAccuracy.txt"
save_txt(accPath,str(test_accuracy_value))
print(test_accuracy_value)


