# Maria Ali
# Machine Learning Assignment 3
# Using Decision Tree Regression to predict which country is going to provide best profit in future

# Importing the libraries and the dataset for evaluation
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
# Adding Columns "R&D Spend, Administration % Marketing Spend"
dataset['Sum'] =  dataset[['R&D Spend', 'Administration', 'Marketing Spend']].sum(axis=1)
                                      # For California
# Assembling the values for "p" and "q":
p = dataset.loc[(dataset.State == 'California'), ['Sum']]
q = dataset.loc[(dataset.State == 'California'), ['Profit']]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
p_train, p_test, q_train, q_test = train_test_split(p, q, test_size = 0.2, random_state = 0)
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(p, q)
# Visualising the Decision Tree Regression results
plt.scatter(p, q, color = 'brown') # Creating Scatter Plot
plt.plot(p, regressor.predict(p), color = 'orange', label='Best Fit Line') # Creating the Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Decision Tree Regression for California',**tnrfont) # Setting the Title
plt.xlabel('The sum of R & D Spend, Administration and Marketing Spend',**tnrfont) # Labelling x-axis
plt.ylabel('Profit',**tnrfont) # Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()
                                        # For Florida
  # Assembling the values for "m" and "n":
m = dataset.loc[(dataset.State == 'Florida'), ['Sum']]
n = dataset.loc[(dataset.State == 'Florida'), ['Profit']]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
m_train, m_test, n_train, n_test = train_test_split(m, n, test_size = 0.2, random_state = 0)
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(m, n)
# Visualising the Decision Tree Regression results
plt.scatter(m, n, color = 'brown') # Creating Scatter Plot
plt.plot(m, regressor.predict(m), color = 'orange', label='Best Fit Line') # Creating the Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Decision Tree Regression for Florida',**tnrfont) # Setting the Title
plt.xlabel('The sum of R & D Spend, Administration and Marketing Spend',**tnrfont) # Labelling x-axis
plt.ylabel('Profit',**tnrfont) # Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()                  