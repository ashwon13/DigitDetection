#Imports
import pandas as pd #Used to read excel file
import numpy as np #Makes matrix algebra much much much easier
import math #Used in activation function (RELU,Sigmoid, Softmax)
import matplotlib.cm as cm #Used to plot data
import matplotlib.pyplot as plt

#Reading in data

train_data = pd.read_csv("train.csv")
test_data= pd.read_csv("test.csv")

train_labels=np.array(train_data.loc[:,'label']) # if you do not understand exactly how to parse this 
#data and how we got only the labels i got you this may help you https://stackoverflow.com/questions/509211/understanding-slice-notation
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
#This article also goes over slicing in pandas very deeply https://www.marsja.se/how-to-use-iloc-and-loc-for-indexing-and-slicing-pandas-dataframes/
#loc for parsing by labels/name iloc for parsing by indexeses
# the first : means select all rows the , sperates the rows from the columns meaning we're done
#slicing the rows then we select only the column with the name 'label' if we wanted to get from 
#'label' and '1x1' we would do 'label'L'1x1' follows start:stop format
print(train_labels.shape)
train_data=np.array(train_data.loc[:,train_data.columns!='label'])#this is some cool logic but
#it's basically selecting all columns without the label name 'label could also do ,'1x1':'28x28' but that would be fixed/rigid and garbage

index=7;
plt.title((train_labels[index])) # gets what the number is suppoused to be using labels column
plt.imshow(train_data[index].reshape(28,28), cmap=cm.binary) #gets the rest of the array which is essentially a 1x785 matrix that
#after using the reshpae command makes it a 28x28 matrix. we then can set the colour map (cmap) to whatever we want
#  
#column 
plt.show()#shows the graph
print("train data")#Werid print statement
y_value=np.zeros((1,10))#creates a 1x10 array of zeros this is 
print(y_value)
for i in range (10):
    print("occurance of ",i,"=",np.count_nonzero(train_labels==i)) # print number of occurences with a method
    #np.count_nonzero method
    y_value[0,i-1]= np.count_nonzero(train_labels==i)#Stores the count of number of times the images equal the certian digit eg 4000 0s 3600 1s 2898 2s


y_value=y_value.ravel()
x_value=[0,1,2,3,4,5,6,7,8,9]
plt.xlabel('label')
plt.ylabel('count')
plt.bar(x_value,y_value,0.7,color='g')#0.7 is width between bars
plt.show()


print(train_data.shape) #(42000, 784)
train_data=np.reshape(train_data,[784,42000]) #don't be an idiot like me all this line of code does is
#go from hab=ving all the pixel values for image be 753x1 to 1x753 by switching the rows with the columns to make a 
#1x753 vector
print(train_data)
#VECTORIZE IT
train_label=np.zeros((10,42000))#creates basically a vector (filled with temp zeros) where you have 10 columns which
#act like a vector of labels where basically the nth column will be one if the image is number n. nut right now its filled with zeros
# it has 10 rows and 42000 columns the columns are all data points. 
for col in range (42000):#there are 42000 columns these are all the images and all these images need to be labeled
    val=train_labels[col]#gets the value of the label in it's corresponding column (train labels contains all the labels in a 42000 x 1 matrix  ) so when you cal train_Labels[col]
    #you get the value of the label in the correspongin column
    for row in range (10):#goes into the the row of the column(still filled with zero) if the row number equals the column then
        if (val==row):
            train_label[val,col]=1 # then set it to one
# TWO DISTINCTIONS TRAIN_LABELS WITH A S LABELS IS 42000 x 1 but TRAIN_LABEL NO S is basically a vector 10x42,000 eg if the label was 3 it would look like
#0   if this is one it means the image is a 0
#0   if this is one it means the image is a 1
#0
#1
#0
#0
#0
#0
#0
#0
#this is one column it goes and their 42,000 columns standing for 42,000 images
print("train_data shape="+str(np.shape(train_data))) #gets size of data
print("train_label shape="+str(np.shape(train_label))) #gets size of label which should be 10x42,000








