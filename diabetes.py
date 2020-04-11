from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.svm import SVC
from math import log2

def read_data():
    data = pd.read_csv('diabetes.csv')
    # finding the mean values
    data.fillna(data.mean(), inplace=True)
    return data

def eda(data):

    ## correlation matrix
    # corr_matrix = data.corr()
    # print(corr_matrix["Outcome"].sort_values(ascending=False))

    #null values
    data[['Glucose','BloodPressure','SkinThickness','Insulin', \
        'BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    # feature_wise_null_count = data.isnull().sum()
    # print(feature_wise_null_count)

    


def visualize(data):
    # data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())
    # data['BMI']=data['BMI'].replace(0,data['BMI'].mean())
    data.plot(kind="scatter",x="Glucose",y="BMI", c = "Outcome", cmap = plt.get_cmap("rainbow"),colorbar=True, alpha=0.6)
    # data.plot(kind="scatter",x="Glucose",y="Outcome")
    plt.show()

def apply_ML(data,f1=None,f2=None,clf_name=None):
    y = np.array(data['Outcome'])
    # X = np.array(data.drop(['Outcome'],axis=1))
    X = np.array(data[[f1,f2]])
    X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
    if clf_name == 'DT':
        clf = DecisionTreeClassifier(max_depth=4,min_samples_leaf=10, criterion='entropy')
    if clf_name == 'GBC':
        clf = GradientBoostingClassifier(n_estimators=2000)
    if clf_name == 'RF':
        clf = RandomForestClassifier()
    if clf_name == 'LR':
        clf = LogisticRegression(C=1e5)
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=50)
    if clf_name == 'SVC':
        clf = SVC(kernel='poly',degree=3, gamma='auto')
    if clf_name == 'SVC_R':
        clf = SVC(kernel='rbf',gamma=0.1,C=1)
    clf.fit(X_tr,y_tr)
    print('Training accuracy:',clf.score(X_tr,y_tr))
    print('Val accuracy:',clf.score(X_val,y_val))
    new_point = np.array([0.1,18]).reshape(1,-1)
    print('Class of new point',clf.predict(new_point))
    if clf_name == 'DT':
        export_graphviz(clf,out_file='tree.dot',filled=True,feature_names=[f1,f2])
    plot_decision_regions(X_tr, y_tr, clf=clf, legend=2)#,  X_highlight=X_val)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title('Classifier')
    plt.show()


def calculate_entropy(n1,n2):
    all_data = n1+n2
    p1 = n1/all_data
    p2 = n2/all_data
    entropy = -(p1*log2(p1) + p2*log2(p2))
    print(entropy)

def c_gini(n1,n2):
    all_data = n1+n2
    p1 = n1/all_data
    p2 = n2/all_data
    gini_score = 1 - p1*p1 - p2*p2
    print(gini_score)

if __name__ == '__main__':
    data = read_data()
    visualize(data)
    apply_ML(data,f1='Glucose',f2='BMI',clf_name='DT')
    # calculate_entropy(0.1,99.9)