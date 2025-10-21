import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {"Study Hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "Score": [35, 50, 55, 70, 65, 75, 85, 90]}
df=pd.DataFrame(data)
print(df)

X=df[["Study Hours"]]
y=df[["Score"]]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4, random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

predictions= model.predict(X_test)
print("Predictions:", predictions)

print("Actual:", y_test.values)
plt.scatter(X_test, y_test, color="red", label="Actual")
plt.plot(X_test, predictions, color="blue", label="Prediction")
plt.legend()
plt.show()