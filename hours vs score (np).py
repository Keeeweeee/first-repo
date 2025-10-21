import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]).reshape(-1, 1)
y = np.array([30, 45, 50, 56, 67, 71, 73, 74, 76, 78, 80, 80, 80, 80])

model1 = LinearRegression()
model1.fit(X, y)

predictions = model1.predict(X)

plt.scatter(X, y, color='grey', label='Original data1')
plt.plot(X, predictions, color='black', label='Regression line1')


print("Model 1 slope:", model1.coef_, "intercept:", model1.intercept_)
print("Prediction for 20 study hours (Model 1):", model1.predict([[20]]))

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([5, 7, 7, 10, 13, 17, 18, 21, 23, 25])

model2 = LinearRegression()
model2.fit(X, y)

predictions = model2.predict(X)

plt.scatter(X, y, color='pink', label='Original data2')
plt.plot(X, predictions, color='blue', label='Regression line2')


print("Model 2 slope:", model2.coef_, "intercept:", model2.intercept_)
print("Prediction for 20 study hours (Model 2):", model2.predict([[20]]))


plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.show()