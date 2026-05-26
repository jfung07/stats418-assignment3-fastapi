import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import pickle
from pathlib import Path

# Part 1: Modeling Requirements

# part a: load mtcars.csv 
base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
filepath_csv = os.path.join(base, "mtcars.csv")
df = pd.read_csv(filepath_csv)

# train model
# part c: use at least one predictor variable
X = df[['cyl', 'hp', 'wt', 'gear', 'am']]
# part b: use mpg as the response
y = df["mpg"]
# part d: train a regression model in python
model = LinearRegression()
model.fit(X, y)

# part e: save trained model to disk
Path("models").mkdir(exist_ok = True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
