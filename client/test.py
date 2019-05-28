import os

os.system("python mnist.py ./result/grad" + str(1) + ".json ./data/data" + str(1) + ".json")
with open("./accuracy/testAccuracy.txt","r") as f:
    text = f.read();
arr = text.split(" ")
print(arr)