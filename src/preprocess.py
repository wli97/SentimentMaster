import os
import math
from sklearn.model_selection import train_test_split

y = [1]*12500
y.append([0]*12500)
x = []

count = 0
for filename in os.listdir("../src/train/pos"):
    with open("./train/pos/"+filename, encoding='utf-8', mode = 'r') as f:
        data = f.read().lower()
        x.append(data)

for filename in os.listdir("../src/train/neg"):
    with open("./train/neg/"+filename, encoding='utf-8', mode = 'r') as f:
        data = f.read().lower()
        x.append(data)
 
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
