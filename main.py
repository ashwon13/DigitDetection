#Imports
import pandas as pd #Used to read excel file
import numpy as np #Makes matrix algebra much much much easier
import math #Used in activation function (RELU,Sigmoid, Softmax)
import matplotlib.cm as cm #Used to plot data
import matplotlib.pyplot as plt

#Reading in data

train_data = pd.read_csv("mnist_train.csv")
test_data= pd.read_csv("mnist_test.csv")

train_labels=np.array(train_data.loc[:,'label']) # if you do not understand exactly how to parse this 
#data and how we got only the labels i got you this may help you https://stackoverflow.com/questions/509211/understanding-slice-notation
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
#This article also goes over slicing in pandas very deeply https://www.marsja.se/how-to-use-iloc-and-loc-for-indexing-and-slicing-pandas-dataframes/
#loc for parsing by labels/name iloc for parsing by indexeses
# the first : means select all rows the , sperates the rows from the columns meaning we're done
#slicing the rows then we select only the column with the name 'label' if we wanted to get from 
#'label' and '1x1' we would do 'label'L'1x1' follows start:stop format
train_data=np.array(train_data.loc[:,train_data.columns!='label'])#this is some cool logic but
#it's basically selecting all columns without the label name 'label could also do ,'1x1':'28x28' but that would be fixed/rigid and garbage

index=7;
plt.title((train_labels[index])) # gets what the number is suppoused to be using labels column
plt.imshow(train_data[index].reshape(28,28), cmap=cm.inferno ) #gets the rest of the array which is essentially a 1x785 matrix that
#after using the reshpae command makes it a 28x28 matrix. we then can set the colour map (cmap) to whatever we want
#  
#column 
plt.show()#shows the graph
print("train data")#Werid print statement
y_value=np.zeros((1,10))#creates a 1x10 array of zeros
print(y_value)
for i in range (10):
    print("occurance of ",i,"=",np.count_nonzero(train_labels==i)) # print number of occurences with a method
    #np.count_nonzero method
    y_value[0,i-1]= np.count_nonzero(train_labels==i)#Stores the count of
# xls = pd.ExcelFile('ex3d1.xlsx')
# df = pd.read_excel(xls, 'X', header = None)

# y = pd.read_excel(xls, 'y', header=None)

# df.shape
# y.shape
# hidden_layer = 25

# y_arr = y[0].unique()#Output:
# array([10,  1,  2,  3,  4,  5,  6,  7,  8,  9], dtype=int64)







