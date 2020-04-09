# Maria Ali
# Machine Learning Assignment 3
# Using Polynomial Regression for future analysis where when ID is inserted the housing price is displayed

# Importing the libraries and the dataset for evaluation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dataset = pd.read_csv('housing price.csv')

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

# Visualising the Linear Regression results
plt.scatter(p, q, color = 'orange') # Creating Scatter Plot
plt.plot(p, lin_reg.predict(p), color = 'brown', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Linear Regression for ID vs. Housing Price',**tnrfont) # Setting the Title
plt.xlabel('ID',**tnrfont) # Labelling x-axis
plt.ylabel('Housing Price',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the Housing Price for Id numbers: 2920, 2925 and 2930 with Linear Regression
hp2920 = lin_reg.predict([[2920]])
hp2925 = lin_reg.predict([[2925]])
hp2930 = lin_reg.predict([[2930]])
print ('The Housing Price for ID number 2920 will be:')
print (hp2920)
print ('The Housing Price for ID number 2925 will be:')
print (hp2925)
print ('The Housing Price for ID number 2930 will be:')
print (hp2930)

# Visualising the Polynomial Regression results
p_grid = np.arange(min(p), max(p), 0.1)
p_grid = p_grid.reshape((len(p_grid), 1))
plt.scatter(p, q, color = 'orange') # Creating Scatter Plot
plt.plot(p_grid, lin_reg_2.predict(poly_reg.fit_transform(p_grid)), color = 'brown', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Polynomial Regression for ID vs. Housing Price',**tnrfont) # Setting the Title
plt.xlabel('ID',**tnrfont) # Labelling x-axis
plt.ylabel('Housing Price',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the Housing Price for Id numbers: 2920, 2925 and 2930 with Polynomial Regression
hp_2920 = lin_reg.predict([[2920]])
hp_2925 = lin_reg.predict([[2925]])
hp_2930 = lin_reg.predict([[2930]])
print ('The Housing Price for ID number 2920 will be:')
print (hp_2920)
print ('The Housing Price for ID number 2925 will be:')
print (hp_2925)
print ('The Housing Price for ID number 2930 will be:')
print (hp_2930)