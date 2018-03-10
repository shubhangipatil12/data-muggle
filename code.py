import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm2
import numpy as np
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import svm

df = pd.read_csv("Data.csv")

years=df.groupby('Year').groups.keys()
country=df.groupby('Country').groups.keys()
a=[]
ylim=2035

for i in range(0,len(country)):
    for j in years:
        a.append(([i,j]))
fp=open("foodproduction.txt",'w')
for j in range(0,len(country)):
    allp=[]
    prod=[]
    allpr=[]
    year=2015
    b=[]
    for i in years:
        all1=df[(df['Country']==country[j])&(df['Year']==i)]
        b.append([j,i])
        prod=all1['Food Production'].values
        allp.append(prod)
    for i in allp:
        allpr.append(i[0])
    while year<=ylim:
        X = np.array(b)
        Y = np.array(allpr)
        clf = LinearRegression()
        clf.fit(X,Y)
        pred=clf.predict([j, year])
        fp.write("\n"+country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print(country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print b,allpr
        year+=1

fp.close()

fp=open("foodconsumption.txt",'w')
for j in range(0,len(country)):
    allp=[]
    prod=[]
    allpr=[]
    year=2015
    b=[]
    for i in years:
        all1=df[(df['Country']==country[j])&(df['Year']==i)]
        b.append([j,i])
        prod=all1['Food Consumption'].values
        allp.append(prod)
    for i in allp:
        allpr.append(i[0])
    while year<=ylim:
        X = np.array(b)
        Y = np.array(allpr)
        clf = LinearRegression()
        clf.fit(X,Y)
        pred=clf.predict([j, year])
        fp.write("\n"+country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print(country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print b,allpr
        year+=1

fp.close()

fp=open("Population.txt",'w')
for j in range(0,len(country)):
    allp=[]
    prod=[]
    allpr=[]
    year=2015
    b=[]
    for i in years:
        all1=df[(df['Country']==country[j])&(df['Year']==i)]
        b.append([j,i])
        prod=all1['Population'].values
        allp.append(prod)
    for i in allp:
        allpr.append(i[0])
    while year<=ylim:
        X = np.array(b)
        Y = np.array(allpr)
        clf = LinearRegression()
        clf.fit(X,Y)
        pred=clf.predict([j, year])
        fp.write("\n"+country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print(country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print b,allpr
        year+=1

fp.close()



fp=open("Wastage.txt",'w')
for j in range(0,len(country)):
    allp=[]
    prod=[]
    allpr=[]
    year=2015
    b=[]
    for i in years:
        all1=df[(df['Country']==country[j])&(df['Year']==i)]
        b.append([j,i])
        prod=all1['Wastage'].values
        allp.append(prod)
    for i in allp:
        allpr.append(i[0])
    while year<=ylim:
        X = np.array(b)
        Y = np.array(allpr)
        clf = LinearRegression()
        clf.fit(X,Y)
        pred=clf.predict([j, year])
        fp.write("\n"+country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print(country[j]+" , "+str(year)+" : "+str(pred[0]))
        #print b,allpr
        year+=1

fp.close()
        
