# Maria Ali
# Machine Learning Assignment 3
# Regression on the Data of monthly experience and income distribution of different employees

# Importing the libraries and the dataset for evaluation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dataset = pd.read_csv('monthlyexp vs incom.csv')

# Assembling the values for "p" and "q":
p = dataset.iloc[:, 0:1].values
q = dataset.iloc[:, 1:2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
p_train, p_test, q_train, q_test = train_test_split(p, q, test_size = 0.2, random_state = 0)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(p, q)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
p_poly = poly_reg.fit_transform(p)
poly_reg.fit(p_poly, q)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(p_poly, q)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(p, q)

# Visualising the Linear Regression results
plt.scatter(p, q, color = 'orange') # Creating Scatter Plot
plt.plot(p, lin_reg.predict(p), color = 'brown', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Linear Regression for Monthly Experience vs. Income Distribution',**tnrfont) # Setting the Title
plt.xlabel('Monthly Experience',**tnrfont) # Labelling x-axis
plt.ylabel('Income Distribution',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Visualising the Polynomial Regression results
p_grid = np.arange(min(p), max(p), 0.1)
p_grid = p_grid.reshape((len(p_grid), 1))
plt.scatter(p, q, color = 'orange') # Creating Scatter Plot
plt.plot(p_grid, lin_reg_2.predict(poly_reg.fit_transform(p_grid)), color = 'brown', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Polynomial Regression for Monthly Experience vs. Income Distribution',**tnrfont) # Setting the Title
plt.xlabel('Monthly Experience',**tnrfont) # Labelling x-axis
plt.ylabel('Income Distribution',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Visualising the Decision Tree Regression results
p_grid = np.arange(min(p), max(q), 0.01)
p_grid = p_grid.reshape((len(p_grid), 1))
plt.scatter(p, q, color = 'orange') # Creating Scatter Plot
plt.plot(p_grid, regressor.predict(p_grid), color = 'brown', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Decision Tree Regression for Monthly Experience vs. Income Distribution',**tnrfont) # Setting the Title
plt.xlabel('Monthly Experience',**tnrfont) # Labelling x-axis
plt.ylabel('Income Distribution',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()