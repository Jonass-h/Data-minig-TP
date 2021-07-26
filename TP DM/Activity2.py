import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#1 chargement du dataset
data=pd.read_csv("<path iris csv file>")
print(data.head(10))

#2 description du dataset

data.info()

#3 Distribution des classe pour les differents  attributs
# sepallength
tmp = data.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='Species', markers='+')

plt.show()

#4 Split the dataset to 2 halves
print("###############")
trainData, testData, = data[0:75], data[75:150]
print(trainData.shape)
print(testData.shape)
#5 training
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)

k=10
knn = KNeighborsClassifier(k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#6 calculate the different metrics

score=metrics.accuracy_score(y_test, y_pred)
confusionMatrix=confusion_matrix(y_test, y_pred)
fp=0
#for the last class
for i in range(0,2):
        fp += confusionMatrix[2][i]

print(confusionMatrix)
tp=confusionMatrix[2][2]
precision=tp/(float)(tp+fp)
print("precision =", precision)
print(score)

