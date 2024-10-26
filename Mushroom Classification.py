#importing the packages

import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import matplotlib.pyplot as plt

#uploading the data in the path

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("mushroon_12.csv")

data.head()

#EDA Part
#Checking the null records
data.isnull().sum()

#checking the duplicates values in the dataset
data[data.duplicated()]

#Removing the duplicated values
data.drop_duplicates(inplace=True)


# To show Outliers in the data set run the code 

num_vars = data.select_dtypes(include=['int','float']).columns.tolist()

num_cols = len(num_vars)
num_rows = (num_cols + 2 ) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate (num_vars):
    sns.boxplot(x=data[var],ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
  for i in range(num_cols , len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()




def zohaib (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["cap-diameter"])
data = zohaib(data,"cap-diameter")

data.boxplot(column=["stem-width"])
data = zohaib(data,"stem-width")


#Checking the distribution of the data
from sklearn.utils import resample
data['class'].value_counts()

#Auto EDA using autoviz package

from autoviz.AutoViz_Class import AutoViz_Class 

AV = AutoViz_Class()

import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'mushroon_12.csv'
sep =","
dft = AV.AutoViz(
    filename  
)


from sklearn import preprocessing
for col in data.select_dtypes(include=['object']).columns:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(data[col].unique())
    data[col] = label_encoder.transform(data[col])
    print(f'{col} : {data[col].unique()}')




#Scale the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Segregating into dependent and independent variable

X = data.drop("class", axis = 1)

y = data["class"]

X.head()

y.head()


#Training and Testing the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Building and Running Different Algorithm
#KNN

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))#confusion matrix
print(classification_report(y_test, y_pred))#full report

print("Training data accuracy:", classifier.score(X_train, y_train))
print("Testing data accuracy", classifier.score(X_test, y_test))



#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
#Testing the model
y_predict_test = classifier.predict(X_test)
#Checking confusion matrix and accuracy of the model
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_predict_test,y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict_test,))
from sklearn.metrics import accuracy_score
predictions = classifier.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')



#Random forest
n_est=100
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=0)
rfc.fit(X_train, y_train);


from sklearn.metrics import accuracy_score
predictions = rfc.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')

print("Training data accuracy:", rfc.score(X_train, y_train))
print("Testing data accuracy", rfc.score(X_test, y_test)) 




#XGboost
from xgboost import XGBClassifier

xgbc = XGBClassifier(tree_method='auto', n_estimators=n_est, n_jobs=-1, random_state=0)
xgbc.fit(X_train, y_train);

print("Training data accuracy:", xgbc.score(X_train, y_train))
print("Testing data accuracy", xgbc.score(X_test, y_test))

from sklearn.metrics import accuracy_score
predictions = xgbc.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')


#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

sgbc = GradientBoostingClassifier(n_estimators=n_est, random_state=0)
sgbc.fit(X_train, y_train);

print("Training data accuracy:", sgbc.score(X_train, y_train))
print("Testing data accuracy", sgbc.score(X_test, y_test))

from sklearn.metrics import accuracy_score
predictions = sgbc.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')



#Bagging Classifier
from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(n_estimators=n_est, random_state=0)
bc.fit(X_train, y_train);

print("Training data accuracy:", bc.score(X_train, y_train))
print("Testing data accuracy", bc.score(X_test, y_test))
from sklearn.metrics import accuracy_score
predictions = bc.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')




#Stacks of Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=n_est, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42, max_iter=1000)))
]
sc = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
sc.fit(X_train, y_train);


print("Training data accuracy:", sc.score(X_train, y_train))
print("Testing data accuracy", sc.score(X_test, y_test))


#ADABOOST

from sklearn.ensemble import AdaBoostClassifier

adc = AdaBoostClassifier(n_estimators=n_est, random_state=0)
adc.fit(X_train, y_train);

print("Training data accuracy:", adc.score(X_train, y_train))
print("Testing data accuracy", adc.score(X_test, y_test))



#Run VOTING CLASSIFIER

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

clf1 = LogisticRegression(multi_class='multinomial', random_state=1, max_iter=5000)
clf2 = RandomForestClassifier(n_estimators=n_est, random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', n_jobs=-1)
eclf1.fit(X_train, y_train)

eclf2 = VotingClassifier(estimators=[
         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
         voting='soft')
eclf2.fit(X_train, y_train);

print("Training data accuracy:", eclf1.score(X_train, y_train))
print("Testing data accuracy", eclf1.score(X_test, y_test))

print("Training data accuracy:", eclf2.score(X_train, y_train))
print("Testing data accuracy", eclf2.score(X_test, y_test))


#XGboost and Knn worked well with the dataset there was no overfitting and underfitting

#KSstatistic Test
from scipy import stats
rng = np.random.default_rng()
stats.kstest(stats.uniform.rvs(size=100, random_state=rng),stats.norm.cdf)

#Data Normality Test
import numpy as np
from scipy.stats import anderson
np.random.seed(0)
data=np.random.normal(size=100)
anderson(data)

#Data is normally distributed
