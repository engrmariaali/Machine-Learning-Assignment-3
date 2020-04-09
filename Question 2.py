# Maria Ali
# Machine Learning Assignment 3
# Using Polynomial Regression to predict the temperature in 2016 and 2017 using the past data of both industries

# Importing the libraries and the dataset for evaluation
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('annual_temp.csv')
                                     # For GCAG
# Assembling the values for "p" and "q":
p = dataset.loc[(dataset.Source == 'GCAG'), ['Year']]
q = dataset.loc[(dataset.Source == 'GCAG'), ['Mean']]

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
plt.scatter(p, q, color = 'brown') # Creating Scatter Plot
plt.plot(p, lin_reg.predict(p), color = 'orange', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Linear Regression for GCAG',**tnrfont) # Setting the Title
plt.xlabel('Year',**tnrfont) # Labelling x-axis
plt.ylabel('Mean Temperature',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the temperature for 2016 and 2017 with Linear Regression
temperature2016 = lin_reg.predict([[2016]])
temperature2017 = lin_reg.predict([[2017]])
print ('The temperature in 2016 will be:')
print (temperature2016)
print ('The temperature in 2017 will be:')
print (temperature2017)

# Visualising the Polynomial Regression results
plt.scatter(p, q, color = 'brown') # Creating Scatter Plot
plt.plot(p, lin_reg_2.predict(poly_reg.fit_transform(p)), color = 'orange', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Polynomial Regression for GCAG',**tnrfont) # Setting the Title
plt.xlabel('Year',**tnrfont) # Labelling x-axis
plt.ylabel('Mean Temperature',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the temperature for 2016 and 2017 with Polynomial Regression
temperaturee2016 = lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
temperaturee2017 = lin_reg_2.predict(poly_reg.fit_transform([[2017]]))
print ('The temperature in 2016 will be:')
print (temperaturee2016)
print ('The temperature in 2017 will be:')
print (temperaturee2017)

                                         # For GISTEMP
# Assembling the values for "m" and "n":
m = dataset.loc[(dataset.Source == 'GISTEMP'), ['Year']]
n = dataset.loc[(dataset.Source == 'GISTEMP'), ['Mean']]

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
m_train, m_test, n_train, n_test = train_test_split(m, n, test_size = 0.2, random_state = 0)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(m, n)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
m_poly = poly_reg.fit_transform(m)
poly_reg.fit(m_poly, n)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(m_poly, n)

# Visualising the Linear Regression results
plt.scatter(m, n, color = 'brown') # Creating Scatter Plot
plt.plot(m, lin_reg.predict(m), color = 'orange', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Linear Regression for GISTEMP',**tnrfont) # Setting the Title
plt.xlabel('Year',**tnrfont) # Labelling x-axis
plt.ylabel('Mean Temperature',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the temperature for 2016 and 2017 with Linear Regression
temperatureee2016 = lin_reg.predict([[2016]])
temperatureee2017 = lin_reg.predict([[2017]])
print ('The temperature in 2016 will be:')
print (temperatureee2016)
print ('The temperature in 2017 will be:')
print (temperatureee2017)

# Visualising the Polynomial Regression results
plt.scatter(m, n, color = 'brown') # Creating Scatter Plot
plt.plot(m, lin_reg_2.predict(poly_reg.fit_transform(m)), color = 'orange', label='Best Fit Line') # Creating Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Polynomial Regression for GISTEMP',**tnrfont) # Setting the Title
plt.xlabel('Year',**tnrfont) # Labelling x-axis
plt.ylabel('Mean Temperature',**tnrfont) #Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Predicting the temperature for 2016 and 2017 with Polynomial Regression
temperatureeee2016 = lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
temperatureeee2017 = lin_reg_2.predict(poly_reg.fit_transform([[2017]]))
print ('The temperature in 2016 will be:')
print (temperatureeee2016)
print ('The temperature in 2017 will be:')
print (temperatureeee2017)