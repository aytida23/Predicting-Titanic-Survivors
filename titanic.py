#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

#Training (891 entries) and Testing (418 entries) data

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = [train_data, test_data]
passenger_id = test_data['PassengerId']

#Feature 1: Pclass
#This shows 1st class passenger survived much more in percentage ratio as compare to the other Pclasses

print(train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean())

#Visualising the total percentage of Pclass

fig = plt.figure(figsize=(18,6))
plt.subplot2grid((2,3), (0,0))
total_pclasses = train_data.Pclass.value_counts()
print(total_pclasses) #there are total 491 passengers of 3rd Pclass, 216 of 1st Pclass, and 184 of 2nd Pclass

train_data.Pclass.value_counts(normalize=True).plot(kind='bar', alpha=0.5) #this plots the bar graph of all Pclasses

#Feature 2: Sex
#This shows the females survived much more as compare to the males by showing the survived percentage ratio

print(train_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean())

# Shows the total number of males and females in the train data set of the ship
fig = plt.figure(figsize=(18,6))
plt.subplot2grid((2,3), (0,0))
total_sex = train_data.Sex.value_counts()
print(total_sex) #this shows there are total 577 males and 314 females
train_data.Sex.value_counts(normalize=True).plot(kind='bar', alpha=0.5) #this plots the bar graph of total number of sex

#Feature 3: Family Size (contain SibSp, Parch and the passenger himself/herself)

for data in all_data:
    data['family_size'] = data['SibSp'] + data['Parch'] + 1
print(train_data[["family_size","Survived"]].groupby(["family_size"], as_index = False).mean())

#shows the total number of family size in the train data set of ship
fig = plt.figure(figsize=(18,6))
plt.subplot2grid((2,3), (0,0))
total_family_size = train_data.family_size.value_counts()
print(total_family_size)
train_data.family_size.value_counts(normalize=True).plot(kind='bar', alpha=0.5)

#Feature 3.1: Is the passenger alone in the ship and survived?
for data in all_data:
    data['is_alone'] = 0
    data.loc[data['family_size'] == 1, 'is_alone'] = 1

print(train_data[['is_alone', 'Survived']].groupby(['is_alone'], as_index = False).mean())

#shows the passenger is alone from the train data set of the ship
fig = plt.figure(figsize=(18,6))
plt.subplot2grid((2,3), (0,0))
is_alone = train_data.is_alone.value_counts()
print(is_alone)
train_data.is_alone.value_counts(normalize=True).plot(kind='bar', alpha=0.5)

#Feature 4: Embarked shows the start point to begin journey of passengers

for data in all_data:
    print(data['Embarked'].value_counts())
    data['Embarked'] = data['Embarked'].fillna('S') #filling na with most occuring value i.e S

print(train_data[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean())

#shows the embarked in a bar graph
fig = plt.figure(figsize=(18,6))
plt.subplot2grid((2,3), (0,0))
train_data.Embarked.value_counts(normalize=True).plot(kind='bar', alpha=0.5)

#Feature 5: Fare
for data in all_data:
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

train_data['category_fare'] = pd.qcut(train_data['Fare'], 4)
print(train_data[['category_fare','Survived']].groupby(['category_fare'], as_index=False).mean())


#shows the category_fare in a bar graph
fig = plt.figure(figsize=(18,6))
plt.subplot2grid((2,3), (0,0))
train_data.category_fare.value_counts(normalize=True).plot(kind='bar', alpha=0.5)

#Feature 6: Age

for data in all_data:
    avg_age = data['Age'].mean()
    age_std = data['Age'].std()
    age_null = data['Age'].isnull().sum()
    
    random_list_age = np.random.randint(avg_age - age_std, avg_age + age_std, size = age_null)
    data['Age'][np.isnan(data['Age'])] = random_list_age
    data['Age'] = data['Age'].astype(int)
    
train_data['category_age'] = pd.qcut(train_data['Age'], 5)
print(train_data[['category_age','Survived']].groupby(['category_age'], as_index = False).mean())


# Feature 7: Name

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

for data in all_data:
    data['title'] = data['Name'].apply(get_title)

for data in all_data:
    data['title'] = data['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['title'] = data['title'].replace('Mlle', 'Miss')
    data['title'] = data['title'].replace('Ms', 'Miss')
    data['title'] = data['title'].replace('Mme', 'Mrs')

print(pd.crosstab(train_data['title'], train_data['Sex']))
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print(train_data[['title', 'Survived']].groupby(['title'], as_index = False).mean())



#Map Data
for data in all_data:

    #Mapping Sex
    sex_map = { 'female':0 , 'male':1 }
    data['Sex'] = data['Sex'].map(sex_map).astype(int)

    #Mapping Title
    title_map = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
    data['title'] = data['title'].map(title_map)
    data['title'] = data['title'].fillna(0)

    #Mapping Embarked
    embark_map = {'S':0, 'C':1, 'Q':2}
    data['Embarked'] = data['Embarked'].map(embark_map).astype(int)

    #Mapping Fare
    data.loc[ data['Fare'] <= 7.91, 'Fare']                            = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare']                               = 3
    data['Fare'] = data['Fare'].astype(int)

    #Mapping Age
    data.loc[ data['Age'] <= 16, 'Age']                       = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age']                        = 4



#Feature Selection
#Create list of columns to drop
drop_elements = ["Name", "Ticket", "Cabin", "SibSp", "Parch", "family_size"]

#Drop columns from both data sets
train_data = train_data.drop(drop_elements, axis = 1)
train_data = train_data.drop(['PassengerId','category_fare', 'category_age'], axis = 1)
test_data = test_data.drop(drop_elements, axis = 1)

#Print ready to use data
print(train_data.head(10))