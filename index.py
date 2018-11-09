# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:19:25 2018

@author: fudu
"""
"                      Step 1: import Proquest data             "
import pandas as pd

A=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\A.csv")
B=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\B.csv")
C=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\C.csv")
D=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\D.csv")
E=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\E.csv")

df=A.append(B)
df=df.append(C)
df=df.append(D)
df=df.append(E)


"                   Step 2: create frequency indices           "

" Separate 'Month' from publication date"
df2=pd.DataFrame(df.pubdate.str.split(' ',1).tolist(),
                                   columns = ['Month',''])
                                  
"Adding month column to proquest dataframe"
df['Month'] = df2['Month']
df=df.sort_values('year')
annual_index=df.groupby('year').count()

" Drop other attributes"
annual_index=annual_index.drop(['pubdate', 'pubtitle', 'Month'],axis=1)


"Create year attribute"
annual_index['Year']=annual_index.index

"Convert annual index to integer"
annual_index=annual_index.astype('int64')

"                Step 3: upload business investment           "

"Upload business investment"
annual_inv=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\annual_business_investment.csv")



"               Step 4: create dataframe with business investment and frequency index "

Correlation['Frequency']=annual_index(['title'],axis=1)

"Correlation"
.corr(method='pearson')



