# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Numpy test
arr = np.array([1, 2, 3, 4, 5])
print("Numpy array:", arr)

# Pandas test
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
print("\nPandas DataFrame:\n", df)

# Matplotlib test
plt.plot(df["x"], df["y"])
plt.title("Test Plot")
plt.show()

# Sklearn test
model = LinearRegression()
model.fit(df[["x"]], df["y"])
print("\nPrediction for x=6:", model.predict([[6]])[0])
# %%
