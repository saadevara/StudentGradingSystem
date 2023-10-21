import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
stud= pd.read_csv('student-mat.csv')    
print('Total number of students:',len(stud))
stud['G3'].describe()
stud.info()   
stud.columns    
stud.describe()    
stud.head()   
stud.tail()    
stud.isnull().any()    



# Scatter plot using matplotlib
plt.scatter(stud['age'], stud['G3'], marker='o', s=8)
plt.xlabel('Age')
plt.ylabel('G3')
plt.title('Scatter plot of Age vs G3')
plt.show()

# Box plot using matplotlib
stud.boxplot(column='G3')
plt.title('Box plot of G3')
plt.show()

# Histogram using matplotlib
plt.hist(stud['G3'], bins=100, color='blue', alpha=0.7)
plt.xlabel('G3')
plt.ylabel('Frequency')
plt.title('Histogram of G3')
plt.show()

sns.heatmap(stud.isnull(),cmap="rainbow",yticklabels=False)    
sns.heatmap(stud.isnull(),cmap="viridis",yticklabels=False)    
f_stud = len(stud[stud['sex'] == 'F'])    
print('Number of female students:',f_stud)
m_stud = len(stud[stud['sex'] == 'M'])    
print('Number of male students:',m_stud)
sns.set_style('whitegrid')    
sns.countplot(x='sex',data=stud,palette='plasma')
b = sns.kdeplot(stud['age'])    
b.axes.set_title('Ages of students')
b.set_xlabel('Age')
b.set_ylabel('Count')
plt.show()
b = sns.countplot(x='age',hue='sex', data=stud, palette='inferno')
b.axes.set_title('Number of Male & Female students in different age groups')
b.set_xlabel("Age")
b.set_ylabel("Count")
plt.show()
u_stud = len(stud[stud['address'] == 'U'])    
print('Number of Urban students:',u_stud)
r_stud = len(stud[stud['address'] == 'R'])    
print('Number of Rural students:',r_stud)
sns.set_style('whitegrid')
sns.countplot(x='address',data=stud,palette='magma')    
sns.countplot(x='address',hue='G3',data=stud,palette='Oranges')
b= sns.boxplot(x='age', y='G3',data=stud,palette='gist_heat')
b.axes.set_title('Age vs Final Grade')
b = sns.swarmplot(x='age', y='G3',hue='sex', data=stud,palette='PiYG')
b.axes.set_title('Does age affect final grade?')
sns.kdeplot(stud.loc[stud['address'] == 'U', 'G3'], label='Urban', shade = True)
sns.kdeplot(stud.loc[stud['address'] == 'R', 'G3'], label='Rural', shade = True)
plt.title('Do urban students score higher than rural students?')
plt.xlabel('Grade');
plt.ylabel('Density')
plt.show()
stud.corr()['G3'].sort_values()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
stud.iloc[:,0]=le.fit_transform(stud.iloc[:,0])
stud.iloc[:,1]=le.fit_transform(stud.iloc[:,1])
stud.iloc[:,3]=le.fit_transform(stud.iloc[:,3])
stud.iloc[:,4]=le.fit_transform(stud.iloc[:,4])
stud.iloc[:,5]=le.fit_transform(stud.iloc[:,5])
stud.iloc[:,8]=le.fit_transform(stud.iloc[:,8])
stud.iloc[:,9]=le.fit_transform(stud.iloc[:,9])
stud.iloc[:,10]=le.fit_transform(stud.iloc[:,10])
stud.iloc[:,11]=le.fit_transform(stud.iloc[:,11])
stud.iloc[:,15]=le.fit_transform(stud.iloc[:,15])
stud.iloc[:,16]=le.fit_transform(stud.iloc[:,16])
stud.iloc[:,17]=le.fit_transform(stud.iloc[:,17])
stud.iloc[:,18]=le.fit_transform(stud.iloc[:,18])
stud.iloc[:,19]=le.fit_transform(stud.iloc[:,19])
stud.iloc[:,20]=le.fit_transform(stud.iloc[:,20])
stud.iloc[:,21]=le.fit_transform(stud.iloc[:,21])
stud.iloc[:,22]=le.fit_transform(stud.iloc[:,22])
stud.head()
stud.tail()
stud.corr()['G3'].sort_values()    
stud = stud.drop(['school', 'G1', 'G2'], axis='columns')
most_correlated = stud.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
most_correlated
stud = stud.loc[:, most_correlated.index]
stud.head()
b = sns.swarmplot(x=stud['failures'],y=stud['G3'],palette='autumn')
b.axes.set_title('Previous Failures vs Final Grade(G3)')
fa_edu = stud['Fedu'] + stud['Medu']
b = sns.swarmplot(x=fa_edu,y=stud['G3'],palette='summer')
b.axes.set_title('Family Education vs Final Grade(G3)')
b = sns.boxplot(x=stud['higher'],y=stud['G3'],palette='binary')
b.axes.set_title('Higher Education vs Final Grade(G3)')
b = sns.countplot(x=stud['goout'],palette='OrRd')
b.axes.set_title('Go Out vs Final Grade(G3)')
b = sns.swarmplot(x=stud['goout'],y=stud['G3'],palette='autumn')
b.axes.set_title('Go Out vs Final Grade(G3)')
b = sns.swarmplot(x=stud['romantic'],y=stud['G3'],palette='YlOrBr')
b.axes.set_title('Romantic Relationship vs Final Grade(G3)')
b = sns.countplot(x='reason',data=stud,palette='gist_rainbow')    
b.axes.set_title('Reason vs Students Count')
b = sns.swarmplot(x='reason', y='G3', data=stud,palette='gist_rainbow')
b.axes.set_title('Reason vs Final grade')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import scipy
X_train, X_test, y_train, y_test = train_test_split(stud, stud['G3'], test_size = 0.25, random_state=42)
X_train.head()
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))
    return mae, rmse
median_pred = X_train['G3'].median()
median_preds = [median_pred for _ in range(len(X_test))]
true = X_test['G3']
mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))



def evaluate(X_train, X_test, y_train, y_test):
    model_name_list = ['Linear Regression', 'Random Forest', 'SVM']
    X_train = X_train.drop('G3', axis='columns')
    X_test = X_test.drop('G3', axis='columns')
    model1 = LinearRegression()
    model2 = RandomForestRegressor(n_estimators=100)
    model3 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)
    for i, model in enumerate([model1, model2, model3]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]
    return results
    joblib.dump(model1, 'linear_regression_model.joblib')

results = evaluate(X_train, X_test, y_train, y_test)

plt.figure(figsize=(12, 7))
ax = plt.subplot(1, 2, 1)
results.sort_values('mae', ascending=True).plot.bar(y='mae', color='violet', ax=ax)
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')

ax = plt.subplot(1, 2, 2)
results.sort_values('rmse', ascending=True).plot.bar(y='rmse', color='pink', ax=ax)
plt.title('Model Root Mean Squared Error')
plt.ylabel('RMSE')

plt.show()


