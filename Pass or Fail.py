import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data={ "Hours studied": [1,2,3,4,5,6,7,8], "Pass": [0,0,0,0,1,1,1,1]}
df=pd.DataFrame(data)

X=df[["Hours studied"]]
y=df[["Pass"]]

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42)

model=LogisticRegression()
model.fit(X_train, y_train)

predictions= model.predict(X_test)
print("Actual:", y_test.values)
print("Prediction:", predictions)

acc=accuracy_score(y_test, predictions)
print("Accuracy:", acc)

plt.scatter(X,y, color="blue", label="Actual Data")
x_range=np.linspace(0,10,100).reshape(-1,1)
plt.plot(x_range, model.predict_proba(x_range)[:,1], color='red', label="Probability of Pass")
plt.xlabel("Hours Studied")
plt.ylabel("Pass Probability")
plt.legend()
plt.show()