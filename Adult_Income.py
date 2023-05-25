
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('adult.csv' , encoding='latin-1')

df.head()

# count the number of rows in the original DataFrame
n_original = df.shape[0]
print(f"{n_original} ")

df['workclass'].replace(to_replace = ['?'], value = 'Other', inplace = True)
df['occupation'].replace(to_replace = ['?'], value = 'Other', inplace = True)
df['native-country'].replace(to_replace = ['?'], value = 'Other', inplace = True)
df.drop_duplicates(inplace=True)

print(df)

df.duplicated()

df.isnull().sum()

df.columns

summary = df.describe()
print(summary)

Q1 = df.quantile(0.25,numeric_only=True)
Q3 = df.quantile(0.75,numeric_only=True)
IQR = Q3 - Q1

# detect outliers in each column
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

# loop through each column and remove outliers
for col in df.columns:
    if outliers[col].any():
        # find the limits
        upper_limit = Q3[col] + 1.5 * IQR[col]
        lower_limit = Q1[col] - 1.5 * IQR[col]
        
        # find the outliers
        outlier_df = df.loc[(df[col] > upper_limit) | (df[col] < lower_limit)]
        print(f"{col} has {len(outlier_df)} outliers")
        
        # remove the outliers
        new_df = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit)]
        df = new_df
        
        # print the number of outliers removed
        print(f"{len(outlier_df)} outliers removed in {col}")



df.shape

dataset = df.copy()

# 1 : Data Exploration
# 1.1
dataset.describe()

dataset['workclass'].value_counts()

dataset['education'].value_counts()

dataset['marital-status'].value_counts()

dataset['occupation'].value_counts()

dataset['relationship'].value_counts()

dataset['race'].value_counts()

dataset['gender'].value_counts()

# 1.2
dataset.info()

#1.3 Summary Statistics For Dataset
# 1.3.1
mean = dataset.mean(numeric_only=True)
print("Mean of the dataset : \n", mean)

# 1.3.2
median = dataset.median(numeric_only=True)
print("Median of the dataset : \n", median)

# 1.3.3
_range = dataset.std(numeric_only=True)
print("Range of the dataset : \n",_range)

# 1.3.4
var = dataset.var(numeric_only=True)
print("Variance of the dataset : \n",var)

# 2 : Visualization Methods

#Workclass Distribution:
    
# 2.1 => Count Plot

init = 1
cp_features = ['workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'gender']

fig = plt.figure(figsize=(25,20))
for feature in cp_features:
    if(init<=6):
        plt.subplot(2, 3, init)
        sns.countplot(data=dataset, x=feature, order=dataset[feature].value_counts().index)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(rotation=45)
        init+=1
    
plt.tight_layout()
plt.show()

#Workclass & Income
sns.countplot(data=dataset, x='workclass', hue='income', 
              order = dataset['workclass'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('')
plt.show()

#Education & Income
sns.countplot(data=dataset, x='education', hue='income', 
              order = dataset['education'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('')
plt.show()

#Occupation & Income
sns.countplot(data=dataset, x='occupation', hue='income', 
              order = dataset['occupation'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('')
plt.show()

# 2.2=> Histogram 
sns.histplot(data=dataset, x='age', hue='income', kde=True)
plt.ylabel('')
plt.xlabel('Age')
plt.show()

# 2.3 => Box Plot
sns.boxplot(data=dataset, x='income', y='age')
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()

#2.4 => Pie Chart
dataset.groupby('gender').size().plot(kind='pie',autopct='%.2f')

# 2.5 => Scatter Plot

plt.scatter(y=dataset['age'],x=dataset['hours-per-week'])

# 2.6 => Scatter Plot Matrix

#The Whole Dataset
sns.pairplot(dataset)
#Educational-num
sns.pairplot(dataset, hue="educational-num")
#Workclass
sns.pairplot(dataset, hue="workclass")

#2.7 => Density Plot

#educational-num Distribution
sns.displot(dataset['educational-num'],color='red',fill='fill',kind='kde')

#hours-per-week Distribution
sns.displot(dataset['hours-per-week'],color='red',fill='fill',kind='kde')

dataset.head()

label_encoder = preprocessing.LabelEncoder()

df['gender'] = label_encoder.fit_transform(df['gender'])
df['workclass'] = label_encoder.fit_transform(df['workclass'])
df['education'] = label_encoder.fit_transform(df['education'])
df['marital-status'] = label_encoder.fit_transform(df['marital-status'])
df['occupation'] = label_encoder.fit_transform(df['occupation'])
df['relationship'] = label_encoder.fit_transform(df['relationship'])
df['native-country'] = label_encoder.fit_transform(df['native-country'])
df['income'] = label_encoder.fit_transform(df['income'])

df.head()

x = df[['age', 'workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'gender', 'hours-per-week', 'native-country']]
y = df['income']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


x.shape, y.shape


print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


knn = KNeighborsClassifier(n_neighbors=9)


knn.fit(x_train, y_train)


predict = knn.predict(x_test)
predict


knn.score(x_test, y_test)*100


print('Accuracy Score: ', accuracy_score(y_test, predict))


print('Precision Score: ', precision_score(y_test, predict))


print('Recall Score: ', recall_score(y_test, predict))


print('F1 Score: ', f1_score(y_test, predict))


print(classification_report(y_test, predict))


cm = confusion_matrix(y_test, predict)
cm



ax = sns.heatmap(cm/np.sum(cm), annot=True,  fmt='.2%', cmap='Blues')

ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

plt.show()



svc=SVC()
svc.fit(x_train, y_train)


svc_predict = svc.predict(x_test)
svc_predict


svc.score(x_test, y_test)*100



print('Accuracy Score: ', accuracy_score(y_test, svc_predict))



svc_cm = confusion_matrix(y_test, svc_predict)
svc_cm



ax = sns.heatmap( svc_cm/np.sum(svc_cm),annot=True,  fmt='.1%',cmap='Greens')

ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

plt.show()



# Random Forest Classifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)



#y_pred = randomforest.predict(x_test)



RF_predict = randomforest.predict(x_test)
RF_predict



score_randomforest = randomforest.score(x_test,y_test)
print('The accuracy of the Random Forest Model is', score_randomforest)


print('Precision Score: ', precision_score(y_test, RF_predict))



print('Recall Score: ', recall_score(y_test, RF_predict))



print('F1 Score: ', f1_score(y_test, RF_predict))



print(classification_report(y_test, RF_predict))



RF_cm = confusion_matrix(y_test, RF_predict)
RF_cm




ax = sns.heatmap( RF_cm/np.sum(RF_cm),annot=True,  fmt='.1%',cmap='Reds')

ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

plt.show()



# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)



# y_pred = gaussian.predict(X_test)
NB_predict = gaussian.predict(x_test)
NB_predict



score_gaussian = gaussian.score(x_test,y_test)
print('The accuracy of Gaussian Naive Bayes is', score_gaussian)



print('Precision Score: ', precision_score(y_test, NB_predict))



print('Recall Score: ', recall_score(y_test, NB_predict))



print('F1 Score: ', f1_score(y_test, NB_predict))


print(classification_report(y_test, NB_predict))



NB_cm = confusion_matrix(y_test, NB_predict)
NB_cm


ax = sns.heatmap( NB_cm/np.sum(NB_cm),annot=True,  fmt='.1%')

ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

plt.show()

