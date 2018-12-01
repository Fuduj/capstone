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
df2=pd.DataFrame(df.pubdate.str.split(' ',1).tolist(),columns = ['Month',''])
                                  
"Adding month column to proquest dataframe"
df['Month'] = df2['Month']
df=df.sort_values('year')
annual_index=df.groupby('year').count()

" Drop other attributes"
annual_index=annual_index.drop(['pubdate', 'pubtitle', 'Month'],axis=1)


"Create year attribute"
annual_index['Year']=annual_index.index

"Convert annual index to integer"
annual_index=annual_index.astype(int)

"Upload total 'Policy' and 'Uncertainty' files "

file_name = 'C:\\Users\\fudu\\Desktop\\Capstone\\EP_articles\\Raw\\{}.csv'
policy = pd.concat([pd.read_csv(file_name.format(i)) for i in range(0, 43)])


"Create one dataframe with total uncertainty/policy count"

policy=policy.sort_values('year')
policy_index=policy.groupby('year').count()
policy_index=policy_index.drop(['pubdate', 'pubtitle'],axis=1)
policy_index['Year']=policy_index.index
policy_index=policy_index[policy_index.index>1963]


"Upload business investment"
annual_inv=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\annual_business_investment.csv").astype(int)
annual_inv.columns=['year','total inv', 'total business inv', 'non-res', 'M&E','IP']
annual_inv.index=annual_inv['year']
annual_inv=annual_inv[annual_inv.index>1963]

        "Step 3: Test for correlations"

"Test contemporaneous correlations between share of uncertainty and measures of BI"

correl_2=pd.DataFrame()
correl_2['EP']=policy_index['Title'].astype(int)
correl_2['EPU']=annual_index['title']
correl_2['share']=(correl_2['EPU']/correl_2['EP'])*100
correl_2 = correl_2.drop(['EP', 'EPU'],axis=1)
correl_2['TI']=annual_inv['total inv']
correl_2['BI']=annual_inv['total business inv']
correl_2['NR']=annual_inv['non-res']
correl_2['ME']=annual_inv['M&E']
correl_2['IP']=annual_inv['IP']

correl_2['TI+1']=annual_inv['total inv'].shift(1)
correl_2['BI+1']=annual_inv['total business inv'].shift(1)
correl_2['NR+1']=annual_inv['non-res'].shift(1)
correl_2['ME+1']=annual_inv['M&E'].shift(1)
correl_2['IP+1']=annual_inv['IP'].shift(1)

correl_2['TI+2']=annual_inv['total inv'].shift(2)
correl_2['BI+2']=annual_inv['total business inv'].shift(2)
correl_2['NR+2']=annual_inv['non-res'].shift(2)
correl_2['ME+2']=annual_inv['M&E'].shift(2)
correl_2['IP+2']=annual_inv['IP'].shift(2)

correl_2['TI+3']=annual_inv['total inv'].shift(3)
correl_2['BI+3']=annual_inv['total business inv'].shift(3)
correl_2['NR+3']=annual_inv['non-res'].shift(3)
correl_2['ME+3']=annual_inv['M&E'].shift(3)
correl_2['IP+3']=annual_inv['IP'].shift(3)

correl_2['TI-1']=annual_inv['total inv'].shift(-1)
correl_2['BI-1']=annual_inv['total business inv'].shift(-1)
correl_2['NR-1']=annual_inv['non-res'].shift(-1)
correl_2['ME-1']=annual_inv['M&E'].shift(-1)
correl_2['IP-1']=annual_inv['IP'].shift(-1)

correl_2['TI-2']=annual_inv['total inv'].shift(-2)
correl_2['BI-2']=annual_inv['total business inv'].shift(-2)
correl_2['NR-2']=annual_inv['non-res'].shift(-2)
correl_2['ME-2']=annual_inv['M&E'].shift(-2)
correl_2['IP-2']=annual_inv['IP'].shift(-2)

correl_2['TI-3']=annual_inv['total inv'].shift(-3)
correl_2['BI-3']=annual_inv['total business inv'].shift(-3)
correl_2['NR-3']=annual_inv['non-res'].shift(-3)
correl_2['ME-3']=annual_inv['M&E'].shift(-3)
correl_2['IP-3']=annual_inv['IP'].shift(-3)

"Check correlations"
correlation=correl_2.corr(method='pearson')

"Results show peak correlations with....."

peak=correlation.sort_values(by=['share'])
peak=peak.drop(['TI','TI+1','TI+2','TI+3','TI-1','TI-2','TI-3','BI','BI+1','BI+2','BI+3','BI-1','BI-2','BI-3','NR','NR-1','NR-2','NR-3','NR+1','NR+2','NR+3','ME','ME+1','ME+2','ME+3','ME-1','ME-2','ME-3','IP','IP+1','IP+2','IP+3','IP-1','IP-2','IP-3'], axis=1)

        "Step 4: Predict business investment with uncertainty"
        
        "1) create forecast model 
        "2) write report
        "3) upload github
        







" test by period"
" predict investment with uncertainty "
" did you understand the topic? did you make enough research? are results competent?"
" show other lagged correlations "

" clarify correlation over timeline " 
" plot investment and frequency share " 
" uncertainty is input, output is investment? "

" which investment variables have biggest impact on uncertainty? most significant?"


