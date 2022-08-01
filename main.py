from sklearn.datasets import load_iris
from engine import KNN
from sklearn.model_selection import train_test_split
from utils import accuracy_score

data = load_iris()
X  = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = KNN()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print('Acc score: ', accuracy_score(pred, y_test))
