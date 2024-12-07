import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn as sk

# k-Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
# Train-Test Split
from sklearn.model_selection import train_test_split
# Accuracy Score
from sklearn.metrics import accuracy_score
# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

#%%

data = pd.read_excel("/Users/sydlo/.spyder-py3/Govt_Units_2024_Final.xlsx")

# Descriptive Statisitcs
# Open Variable "des" for mean & std, ranges printed using max-min

des = data.describe()
print(des)

cen_min = des["CENSUS_ID_PID6"][3]

ranges = []
for i in des:
    ranges.append(des[i][7]-des[i][3])

print(ranges)

#%%

# Search for and describe interesting relationships between columns of categorical data.

pop = data["POPULATION"]
unit = data["UNIT_TYPE"]

unit_nums = []
for i in unit:
    if i == "2 - MUNICIPAL":
        unit_nums.append(2)
    if i == "1 - COUNTY":
        unit_nums.append(1)
    if i == "3 - TOWNSHIP":
        unit_nums.append(3)
        

corr_pop_class = scipy.stats.pearsonr(pop,unit_nums)
print("Correlation Score Between Population and Unit Type: ", corr_pop_class[0])

plt.scatter(unit_nums,pop)
plt.xlabel("Unit Type")
plt.ylabel("Population")
plt.title("Population vs. Unit Type")
plt.show()

pol = data["POLITICAL_CODE_DESCRIPTION"]

pol_nums = []
for i in pol:
    if i == "BOROUGH":
        pol_nums.append(1)
    if i == "CHARTER TOWNSHIP":
        pol_nums.append(2)
    if i == "CITY":
        pol_nums.append(3)
    if i == "COUNTY":
        pol_nums.append(4)
    if i == "METRO TOWNSHIP":
        pol_nums.append(5)
    if i == "METROPOLITAN GOVERNMENT":
        pol_nums.append(6)
    if i == "MUNICIPALITY":
        pol_nums.append(7)
    if i == "PARISH":
        pol_nums.append(8)
    if i == "PLANTATION":
        pol_nums.append(9)
    if i == "TOWN":
        pol_nums.append(10)
    if i == "TOWNSHIP":
        pol_nums.append(11)
    if i == "UNIFIED GOVERNMENT":
        pol_nums.append(12)
    if i == "URBAN COUNTY GOVERNMENT":
        pol_nums.append(13)
    if i == "VILLAGE":
        pol_nums.append(14)
    if i == "METRO GOVERNMENT":
        pol_nums.append(15)
    if i == "CITY AND BOROUGH":
        pol_nums.append(16)
    if i == "CITY AND COUNTY":
        pol_nums.append(17)
    if i == "CITY-PARISH":
        pol_nums.append(18)
    if i == "CIVIL TOWNSHIP":
        pol_nums.append(19)
    if i == "CONSOLIDATED GOVERNMENT":
        pol_nums.append(20)
    if i == "CORPORATION":
        pol_nums.append(21)
    

corr_pop_pol = scipy.stats.pearsonr(pop,pol_nums)
print(corr_pop_pol[0])

plt.scatter(pol_nums,pop)
plt.xlabel("Political Code Description")
plt.ylabel("Population")
plt.title("Population vs. Political Code Description")
plt.show()

# Although there semms to be little to no correlation between population and
# Either Unit Type or Political Code Description, the graphs show that population
# typically remains around the same for every classification and the larger populations
# can be found when observing counties as a whole, or cities.
# This is to be expected as zooming out to the couty level will raise the population count
# and observing urban major cities will also yeild large population numbers due to the
# density in these areas.

#%%

# Chi2 Test

contingency_table = pd.crosstab(data["UNIT_TYPE"],data["POLITICAL_CODE_DESCRIPTION"])
chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)
print("Chi-square statistic: ", chi2)
print("P-value: ", p)
print("Degrees of freedom: ", dof)
print("Expected frequencies: ", expected)

# This test reports a high chi2 statistic. This means we observe a significant difference
# between the observed and expexted data. The expected frequencies are printed out and
# the observed data can be displayed by opening the variable "contingency_table."

#%% 

# Machine Learning

dataML = pd.DataFrame(data["POPULATION"])
dataML['UNIT_TYPE'] = data["UNIT_TYPE"]
dataML["POLITICAL_CODE_DESCRIPTION"] = pol_nums
pd.DataFrame()
x = dataML[["POPULATION","POLITICAL_CODE_DESCRIPTION"]]
y = dataML["UNIT_TYPE"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train, y_train)
    
# Use the Test Set to Evaluate the Model
prediction = classifier.predict(x_test)
#print(prediction[0:100])
#print(y_test[0:100])
    
# Evaluate the Performance of the Model
accuracy = accuracy_score(y_test, prediction)
print('Accuracy Score: {0:.2f} %'.format(accuracy*100))
ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)

# The Machine Learning process works best with 1 nearest neighbor. This was found by 
# Setting up a for loop and running the process with neighbor values from 1-50,
# using every other value.
# The machine has an accuracy of 77 - 78 %, varying with every run. This is fairly
# accurate based on the data provided, as population varies largely throughout the 
# three unit_type classifications.

#%%

# Research Questions

# 1 - Which counties hold the highest and lowest percentage of the 
#     population for each state?

data_mask = data.mask(data['UNIT_TYPE']!='1 - COUNTY')
data_mask = data_mask.dropna()

states = data_mask["STATE"].unique()
keepdata = []

for state in states:
    states_data = []
    statedata = data_mask[data_mask["STATE"]==state]
    units = statedata["UNIT_NAME"].unique()
    for unit in units:
        states_data.append([state,unit,statedata[statedata["UNIT_NAME"]==unit]["POPULATION"].iloc[0]])
        
    keepdata.append(states_data)

cols = ["STATE","COUNTY","POPULATION"]
states_dfs = []

for i in keepdata:
    df = pd.DataFrame(i,columns=cols)
    states_dfs.append(df)
   
states_max_min = []

for i in states_dfs:
    tot_pop = i["POPULATION"].sum()
    max_pop = i["POPULATION"].max()
    min_pop = i["POPULATION"].min()
    name = i["STATE"][0]
    count_max = i[i["POPULATION"]==max_pop]["COUNTY"].iloc[0]
    count_min = i[i["POPULATION"]==min_pop]["COUNTY"].iloc[0]
    
    max_per = (max_pop/tot_pop)*100
    min_per = (min_pop/tot_pop)*100
    
    states_max_min.append([name,count_max,max_per,count_min,min_per])
    
for i in states_max_min:
    print("For {0} the county with the highest density of population is {1} with {2:.2f}%\nthe county with the lowest population is {3} with {4:.2f}%".format(i[0],i[1],i[2],i[3],i[4]))


#%%

# 2 - Based on population and potentially various other variables 
#     can a Machine Learning process correctly predict which type of 
#     classification each data point is?

# Machine Learning

dataML2 = pd.DataFrame(data["POPULATION"])
dataML2['UNIT_TYPE'] = unit_nums
dataML2['ZIP'] = data['ZIP']
dataML2["POLITICAL_CODE_DESCRIPTION"] = pol_nums
dataML2 = dataML2.dropna()
x = dataML2[["POPULATION","UNIT_TYPE","ZIP"]]
y = dataML2["POLITICAL_CODE_DESCRIPTION"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


classifier = KNeighborsClassifier(n_neighbors=13)
classifier.fit(x_train, y_train)
        
# Use the Test Set to Evaluate the Model
prediction = classifier.predict(x_test)
#print(prediction[0:100])
#print(y_test[0:100])
        
# Evaluate the Performance of the Model
accuracy = accuracy_score(y_test, prediction)
print('Accuracy Score: {0:.2f} %'.format(accuracy*100))
ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)

#ML2_corr_matrix = data.corr(numeric_only=True)
#plt.matshow(ML2_corr_matrix)
#plt.colorbar()
#plt.xticks(range(19), data.columns, rotation=90)
#plt.yticks(range(19),data.columns, rotation=0)

