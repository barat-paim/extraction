import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
import ruptures  # pip install ruptures

path = '/Users/btsznh/Downloads/data1.csv'
# read csv file
df = pd.read_csv(path)
#print how many rows and columns
print(df.shape)