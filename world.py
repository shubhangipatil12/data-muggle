import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm2
import numpy as np
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn import svm


df = pd.read_csv("Data.csv")

years=df.groupby('Year').groups.keys()
country=df.groupby('Country').groups.keys()
a=[]
ylim=2035
fp=open("World_Food_Production.txt",'w')
allp=[]
prod=0
allpr=[]
year=2015
b=[]
j=0

for i in years:
    all1=df[(df['Year']==i)]
    b.append([j,i])
    prod=all1['Food Production'].values.sum()
    allp.append(prod)
    j+=1
for i in allp:
    allpr.append(i)
print allpr
while year<=ylim:
    X = np.array(b)
    Y = np.array(allpr)
    clf = LinearRegression()
    clf.fit(X,Y)
    pred=clf.predict([j, year])
    fp.write("\n"+str(year)+" : "+str(pred[0]))
    #print(str(year)+" : "+str(pred[0]))
    #print b,allpr
    year+=1

fp.close()

fp=open("World_Food_Consumption.txt",'w')
allp=[]
prod=0
allpr=[]
year=2015
b=[]
j=0
for i in years:
    all1=df[(df['Year']==i)]
    b.append([j,i])
    prod=all1['Food Consumption'].values.sum()
    allp.append(prod)
    j+=1
for i in allp:
    allpr.append(i)
print allpr
while year<=ylim:
    X = np.array(b)
    Y = np.array(allpr)
    clf = LinearRegression()
    clf.fit(X,Y)
    pred=clf.predict([j, year])
    fp.write("\n"+str(year)+" : "+str(pred[0]))
    #print(str(year)+" : "+str(pred[0]))
    #print b,allpr
    year+=1

fp.close()

fp=open("World_Population.txt",'w')
allp=[]
prod=0
allpr=[]
year=2015
b=[]
j=0
for i in years:
    all1=df[(df['Year']==i)]
    b.append([j,i])
    prod=all1['Population'].values.sum()
    allp.append(prod)
    j+=1
for i in allp:
    allpr.append(i)
print allpr
while year<=ylim:
    X = np.array(b)
    Y = np.array(allpr)
    clf = LinearRegression()
    clf.fit(X,Y)
    pred=clf.predict([j, year])
    fp.write("\n"+str(year)+" : "+str(pred[0]))
    #print(str(year)+" : "+str(pred[0]))
    #print b,allpr
    year+=1

fp.close()

fp=open("World_Wastage.txt",'w')
allp=[]
prod=0
allpr=[]
year=2015
b=[]
j=0
for i in years:
    all1=df[(df['Year']==i)]
    b.append([j,i])
    prod=all1['Wastage'].values.sum()
    allp.append(prod)
    j+=1
for i in allp:
    allpr.append(i)
print allpr
while year<=ylim:
    X = np.array(b)
    Y = np.array(allpr)
    clf = LinearRegression()
    clf.fit(X,Y)
    pred=clf.predict([j, year])
    fp.write("\n"+str(year)+" : "+str(pred[0]))
    #print(str(year)+" : "+str(pred[0]))
    #print b,allpr
    year+=1

fp.close()
