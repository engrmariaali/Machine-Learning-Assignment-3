# Maria Ali
# Machine Learning Assignment 3
# Using Polynomial Regression to predict the CO2 production for the years 2011, 2012 and  2013 using the old data set

# Importing the libraries and the dataset for evaluation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dataset = pd.read_csv('global_co2.csv')

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
plt.title('Linear Regression for Year vs. Carbon dioxide Production',**tnrfont) # Setting the Title
plt.xlabel('Year',**tnrfont) # Labelling x-axis
plt.ylabel('Carbon dioxide Production',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the Carbon dioxide production for years 2011, 2012 and 2013 with Linear Regression
y2011 = lin_reg.predict([[2011]])
y2012 = lin_reg.predict([[2012]])
y2013 = lin_reg.predict([[2013]])
print ('The Carbon dioxide production in 2011 will be:')
print (y2011)
print ('The Carbon dioxide production in 2012 will be:')
print (y2012)
print ('The Carbon dioxide production in 2013 will be:')
print (y2013)

# Visualising the Polynomial Regression results
p_grid = np.arange(min(p), max(p), 0.1)
p_grid = p_grid.reshape((len(p_grid), 1))
plt.scatter(p, q, color = 'orange') # Creating Scatter Plot
plt.plot(p_grid, lin_reg_2.predict(poly_reg.fit_transform(p_grid)), color = 'brown', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Polynomial Regression for Year vs. Carbon dioxide Production',**tnrfont) # Setting the Title
plt.xlabel('Year',**tnrfont) # Labelling x-axis
plt.ylabel('Carbon dioxide Production',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the Carbon dioxide production for years 2011, 2012 and 2013 with Polynomial Regression
y_2011 = lin_reg.predict([[2011]])
y_2012 = lin_reg.predict([[2012]])
y_2013 = lin_reg.predict([[2013]])
print ('The Carbon dioxide production in 2011 will be:')
print (y_2011)
print ('The Carbon dioxide production in 2012 will be:')
print (y_2012)
print ('The Carbon dioxide production in 2013 will be:')
print (y_2013)