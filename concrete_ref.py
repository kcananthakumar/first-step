# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Data Characteristics:
# Compressive strength or compression strength is the capacity of a material or structure to withstand loads tending to reduce size, as opposed to tensile strength, which withstands loads tending to elongate.
# 
# compressive strength is one of the most important engineering properties of concrete. It is a standard industrial practice that the concrete is classified based on grades. This grade is nothing but the Compressive Strength of the concrete cube or cylinder. Cube or Cylinder samples are usually tested under a compression testing machine to obtain the compressive strength of concrete. The test requisites differ country to country based on the design code.
# 
# The concrete compressive strength is a highly nonlinear function of age and ingredients .These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.
#     
# The actual concrete compressive strength (MPa) for a given mixture under a specific age (days) was determined from laboratory. Data is in raw form (not scaled). 
# 
# The compressive strength of concrete can be calculated by the failure load divided with the cross sectional area resisting the load and reported in pounds per square inch in US customary units and mega pascals (MPa) in SI units. Concrete's compressive strength requirements can vary from 2500 psi (17 MPa) for residential concrete to 4000psi (28 MPa) and higher in commercial structures. Higher strengths upto and exceeding 10,000 psi (70 MPa) are specified for certain applications.
# %% [markdown]
# # Attribute information
# * Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
# * Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
# * Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
# * Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
# * Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
# * Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
# * Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
# * Age -- quantitative -- Day (1~365) -- Input Variable
# * Concrete compressive strength -- quantitative -- MPa(megapascals) -- Output Variable

# %%
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Data - first few rows

# %%
data = pd.read_csv(r"concrete.csv")
data.head()


# %%
#renaming columns
data = data.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':"cement",
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':"furnace_slag",
       'Fly Ash (component 3)(kg in a m^3 mixture)':"fly_ash",
       'Water  (component 4)(kg in a m^3 mixture)':"water",
       'Superplasticizer (component 5)(kg in a m^3 mixture)':"super_plasticizer",
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':"coarse_agg",
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':"fine_agg", 'Age (day)':"age",
       'Concrete compressive strength(MPa, megapascals) ':"compressive_strength"})

# %% [markdown]
# # Data Information

# %%
data.info()

# %% [markdown]
# # Missing values

# %%
print (data.isnull().sum())

# %% [markdown]
# # Data dimensions

# %%
print("Number of rows    :",data.shape[0])
print("Number of columns :",data.shape[1])

# %% [markdown]
# # Surface plot for variables
# * X - Axis = columns encoded as [ coarse_agg : 1, fine_agg : 2 , cement : 3 , fly_ash : 4 , water : 5 ,      furnace_slag:6,super_plasticizer : 7 , compressive_strength : 8 , age : 9]
# * Y - Axis = index.
# * Z - Axis = values .

# %%
from mpl_toolkits.mplot3d import Axes3D
data1 = data.copy()

data1 = data1.sort_values(by=['coarse_agg'],ascending=True).reset_index()

data1 = data1[['cement', 'furnace_slag', 'fly_ash', 'water', 'super_plasticizer',
               'coarse_agg', 'fine_agg', 'age', 'compressive_strength']] 

df = data1.unstack().reset_index()
df.columns = ["X","Y","Z"]


df["X"] = df["X"].map({'coarse_agg':1, 'fine_agg':2 , 
                       'cement':3, 'furnace_slag':6, 'fly_ash':4,
                       'water':5, 'super_plasticizer':7,
                        'age':9, 'compressive_strength':8})
 
fig = plt.figure(figsize=(14,9))

ax  = fig.gca(projection = "3d")

surf = ax.plot_trisurf(df["X"],df["Y"],df["Z"],cmap="jet",linewidth=2)
lab  = fig.colorbar(surf,shrink=.5,aspect=5)
lab.set_label("values",fontsize=15)

ax.set_xlabel("variable_encoded")
ax.set_ylabel("index")
ax.set_zlabel("values")

plt.title("Surface plot",color="navy")
plt.show()

# %% [markdown]
# # Variables summary

# %%
plt.figure(figsize=(12,8))
sns.heatmap(round(data.describe()[1:].transpose(),2),linewidth=2,annot=True,fmt="f")
plt.xticks(fontsize=20)
plt.yticks(fontsize=12)
plt.title("Variables summary")
plt.show()

# %% [markdown]
# # Ingredients Distribution

# %%
cols = [i for i in data.columns if i not in 'compressive_strength']
length = len(cols)
cs = ["b","r","g","c","m","k","lime","c"]
fig = plt.figure(figsize=(13,25))

for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(4,2,j+1)
    ax = sns.distplot(data[i],color=k,rug=True)
    ax.set_facecolor("w")
    plt.axvline(data[i].mean(),linestyle="dashed",label="mean",color="k")
    plt.legend(loc="best")
    plt.title(i,color="navy")
    plt.xlabel("")

# %% [markdown]
# # Compressive strength distribution

# %%
plt.figure(figsize=(13,6))
sns.distplot(data["compressive_strength"],color="b",rug=True)
plt.axvline(data["compressive_strength"].mean(),
            linestyle="dashed",color="k",
            label='mean',linewidth=2)
plt.legend(loc="best",prop={"size":14})
plt.title("Compressivee strength distribution")
plt.show()

# %% [markdown]
# # Pair plot between variables

# %%
sns.pairplot(data,markers="h")
plt.show()

# %% [markdown]
# # Contour plot between ingredients and compressive strength

# %%
cols = [i for i in data.columns if i not in 'compressive_strength']
length = len(cols)

plt.figure(figsize=(13,27))
for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(4,2,j+1)
    sns.kdeplot(data[i],
                data["compressive_strength"],
                cmap="hot",
                shade=True)
    plt.title(i+"  &  compressive_strength",color="navy")

# %% [markdown]
# # Correlation between variables

# %%
cor = data.corr()

mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,10))

with sns.axes_style("white"):
    sns.heatmap(cor,annot=True,linewidth=2,
                mask = mask,cmap="magma")
plt.title("Correlation between variables")
plt.show()

# %% [markdown]
# # Swarm plot for variables

# %%

cols = ['cement', 'furnace_slag', 'fly_ash', 'water', 'super_plasticizer',
       'coarse_agg', 'fine_agg', 'age', 'compressive_strength'] 



length = len(cols)

plt.figure(figsize=(12,25))

for i,j in itertools.zip_longest(cols,range(length)):
    
    plt.subplot(3,3,j+1)
    ax = sns.swarmplot( y = data[i],color="orange")
    ax.set_facecolor("k")
    ax.set_ylabel("")
    ax.set_title(i,color="navy")
    plt.subplots_adjust(wspace = .3)

# %% [markdown]
# # 3D plot for cement ,compressive strength and super plasticizer
# * X - Axis = cement.
# * Y - Axis = compressive strength.
# * Z - Axis = super plasticizer.
# * Color    = Age

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14,11))

ax  = fig.gca(projection = "3d")
#plt.subplot(111,projection = "3d") 

plot =  ax.scatter(data["cement"],
           data["compressive_strength"],
           data["super_plasticizer"],
           linewidth=1,edgecolor ="k",
           c=data["age"],s=100,cmap="cool")

ax.set_xlabel("cement")
ax.set_ylabel("compressive_strength")
ax.set_zlabel("super_plasticizer")

lab = fig.colorbar(plot,shrink=.5,aspect=5)
lab.set_label("AGE",fontsize = 15)

plt.title("3D plot for cement,compressive strength and super plasticizer",color="navy")
plt.show()


# %%
#Binning days to months
def label(data):
    if data["age"] <= 30:
        return "1 month"
    if data["age"] > 30 and data["age"] <= 60 :
        return "2 months"
    if data["age"] > 60 and data["age"] <= 90 :
        return "3 months"
    if data["age"] > 90 and data["age"] <= 120 :
        return "4 months"
    if data["age"] > 120 and data["age"] <= 150 :
        return "5 months"
    if data["age"] > 150 and data["age"] <= 180 :
        return "6 months"
    if data["age"] > 180 and data["age"] <= 210 :
        return "7 months"
    if data["age"] > 210 and data["age"] <= 240 :
        return "8 months"
    if data["age"] > 240 and data["age"] <= 270 :
        return "9 months"
    if data["age"] > 270 and data["age"] <= 300 :
        return "10 months"
    if data["age"] > 300 and data["age"] <= 330 :
        return "11 months"
    if data["age"] > 330 :
        return "12 months"
data["age_months"] = data.apply(lambda data:label(data) , axis=1)

# %% [markdown]
# 
# # Age distribution in months

# %%
plt.figure(figsize=(12,5))
order = ['1 month','2 months', '3 months','4 months','6 months','9 months', '12 months']
ax = sns.countplot(data["age_months"],
                   order=order,linewidth=2,
                   edgecolor = "k"*len(order),
                   palette=["w"])
ax.set_facecolor("royalblue")
plt.title("age distribution in months")
plt.grid(True,alpha=.3)
plt.show()

# %% [markdown]
# # Compreesive strength by months

# %%
age_mon = data.groupby("age_months")["compressive_strength"].describe().reset_index()

order  = ['1 month','2 months', '3 months','4 months','6 months','9 months', '12 months']
cols   = [ 'mean', 'std' , 'min' , 'max']
length = len(cols)
cs     = ["b","orange","white","r"] 

fig = plt.figure(figsize=(13,15))

for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(4,1,j+1)
    ax = sns.pointplot("age_months",i,data=age_mon,
                       order=order,
                       markers="H",
                       linestyles="dotted",color=k)
    plt.subplots_adjust(hspace=.5)
    ax.set_facecolor("k")
    plt.title(i+" - compressive strength by months",color="navy")

# %% [markdown]
# # parllell coordinates plot for 3 categories 
# * Concrete's compressive strength requirements can vary from 2500 psi (17 MPa) for residential concrete to 4000psi (28 MPa) and higher in commercial structures. Higher strengths upto and exceeding 10,000 psi (70 MPa) are specified for certain applications.
# * Binning compressive strength in 3 categories.
# * category 1 - mpa  less than 17.
# * category 2 - mpa in between 17 to 28.
# * category 3 - mpa greater than 28.

# %%
data2  =  data.copy()

def lab(data2):
    if data2["compressive_strength"] <= 17:
        return "category 1"
    if data2["compressive_strength"] >17 and data2["compressive_strength"] <= 28 :
        return "category 2"
    if data2["compressive_strength"] >28 :
        return "category 3 "
    
data2["compressive_strength_category"] = data2.apply(lambda data2:lab(data2) ,axis =1)

from pandas.plotting import parallel_coordinates

cols1 = ['cement', 'furnace_slag','fly_ash', 'water',"compressive_strength_category",'age']

plt.figure(figsize=(12,8))
parallel_coordinates(data2[cols1],"compressive_strength_category")
plt.title("parllell coordinates plot for 3 categories")
plt.show()

# %% [markdown]
# # scatter plot between cement and water
# * X - axis = water.
# * Y - axis = cement.
# * SIZE and COLOR = compressive strength.

# %%
fig = plt.figure(figsize=(13,8))
ax = fig.add_subplot(111)
plt.scatter(data["water"],data["cement"],
            c=data["compressive_strength"],s=data["compressive_strength"]*3,
            linewidth=1,edgecolor="k",cmap="viridis")
ax.set_facecolor("w")
ax.set_xlabel("water")
ax.set_ylabel("cement")
lab = plt.colorbar()
lab.set_label("compressive_strength")
plt.title("scatter plot between cement and water")
plt.grid(True,alpha=.3)
plt.show()

# %% [markdown]
# # scatter plot between fine_agg and coarse_agg.
# * X - axis = fine_agg.
# * Y - axis = coarse_agg.
# * SIZE and COLOR = compressive strength.

# %%
fig = plt.figure(figsize=(13,8))
ax = fig.add_subplot(111)
plt.scatter(data["fine_agg"],data["coarse_agg"],
            c=data["compressive_strength"],s=data["compressive_strength"]*4,
            linewidth=1,edgecolor="k",cmap="viridis_r")
ax.set_facecolor("w")
ax.set_xlabel("fine_agg")
ax.set_ylabel("cement")
lab = plt.colorbar()
lab.set_label("compressive_strength")
plt.title("scatter plot between fine_agg and coarse_agg")
plt.grid(True,alpha=.3)
plt.show()

# %% [markdown]
# 
# ### Splitting train and test data

# %%
#Splitting train and test data
from sklearn.model_selection import train_test_split

train,test = train_test_split(data,test_size =.3,random_state = 123)
train_X = train[[x for x in train.columns if x not in ["compressive_strength"] + ["age_months"]]]
train_Y = train["compressive_strength"]
test_X  = test[[x for x in test.columns if x not in ["compressive_strength"] + ["age_months"]]]
test_Y  = test["compressive_strength"]

# %% [markdown]
# # Model

# %%
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

target = "compressive_strength"
def model(algorithm,dtrainx,dtrainy,dtestx,dtesty,of_type):
    
    print (algorithm)
    print ("***************************************************************************")
    algorithm.fit(dtrainx,dtrainy)
    prediction = algorithm.predict(dtestx)
    print ("ROOT MEAN SQUARED ERROR :", np.sqrt(mean_squared_error(dtesty,prediction)) )
    print ("***************************************************************************")
    prediction = pd.DataFrame(prediction)
    cross_val = cross_val_score(algorithm,dtrainx,dtrainy,cv=20,scoring="neg_mean_squared_error")
    cross_val = cross_val.ravel()
    print ("CROSS VALIDATION SCORE")
    print ("************************")
    print ("cv-mean :",cross_val.mean())
    print ("cv-std  :",cross_val.std())
    print ("cv-max  :",cross_val.max())
    print ("cv-min  :",cross_val.min())
    
    plt.figure(figsize=(13,28))
    plt.subplot(211)
    
    testy = dtesty.reset_index()["compressive_strength"]
    
    ax = testy.plot(label="originals",figsize=(12,13),linewidth=2)
    ax = prediction[0].plot(label = "predictions",figsize=(12,13),linewidth=2)
  #  plt.axhline(testy.mean(),color = "r",linestyle="dashed",label=("original_mean:",testy.mean()))
  #  plt.axhline(prediction[0].mean(),color="b",linestyle = "dashed",label=("prediction_mean:",prediction[0].mean()))
    plt.legend(loc="best")
    plt.title("ORIGINALS VS PREDICTIONS")
    plt.xlabel("index")
    plt.ylabel("values")
    ax.set_facecolor("k")
    
    plt.subplot(212)
    
    if of_type == "coef":
        coef = pd.DataFrame(algorithm.coef_.ravel())
        coef["feat"] = dtrainx.columns
        ax1 = sns.barplot(coef["feat"],coef[0],palette="jet_r",
                          linewidth=2,edgecolor="k"*coef["feat"].nunique())
        ax1.set_facecolor("lightgrey")
        ax1.axhline(0,color="k",linewidth=2)
        plt.ylabel("coefficients")
        plt.xlabel("features")
        plt.title('FEATURE IMPORTANCES')
    
    elif of_type == "feat":
        coef = pd.DataFrame(algorithm.feature_importances_)
        coef["feat"] = dtrainx.columns
        ax2 = sns.barplot(coef["feat"],coef[0],palette="jet_r",
                          linewidth=2,edgecolor="k"*coef["feat"].nunique())
        ax2.set_facecolor("lightgrey")
        ax2.axhline(0,color="k",linewidth=2)
        plt.ylabel("coefficients")
        plt.xlabel("features")
        plt.title('FEATURE IMPORTANCES')

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Linear Regression

# %%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model(lr,train_X,train_Y,test_X,test_Y,"coef")

# %% [markdown]
# # Lasso Regression

# %%
from sklearn.linear_model import Ridge,Lasso
ls = Lasso()
model(ls,train_X,train_Y,test_X,test_Y,"coef")

# %% [markdown]
# # Ridge Regression

# %%
rigde = Ridge()
model(rigde,train_X,train_Y,test_X,test_Y,"coef")

# %% [markdown]
# # KNN Regressor

# %%
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(train_X,train_Y)

predictions = knn.predict(test_X)
predictions = pd.DataFrame(predictions)

test_y_new = test_Y.reset_index()
test_y_new = test_y_new["compressive_strength"]


ax3 = test_y_new.plot(label="originals",figsize=(12,6),linewidth=2)
ax3 = predictions[0].plot(label="predictions",figsize=(12,6),linewidth=2)
plt.legend(loc="best")
plt.title("ORIGINALS VS PREDICTIONS")
plt.xlabel("index")
plt.ylabel("values")
ax3.set_facecolor("k")

print (knn)
print ("************************************************************************")
print ("ROOT MEAN SQUARED ERROR : ",np.sqrt(mean_squared_error(test_Y,predictions)))
cross_valid = cross_val_score(knn,train_X,train_Y,cv=20,scoring="neg_mean_squared_error")
cross_valid = cross_valid.ravel()
print ("************************************************************************")
print ("CROSS VALIDATION SCORE")
print ("************************")
print ("cv-mean :",cross_valid.mean())
print ("cv-std  :",cross_valid.std())
print ("cv-max  :",cross_valid.max())
print ("cv-min  :",cross_valid.min())

# %% [markdown]
# # Ada Boost Regressor

# %%
from sklearn.ensemble import AdaBoostRegressor
adb = AdaBoostRegressor()
model(adb,train_X,train_Y,test_X,test_Y,"feat")

# %% [markdown]
# # ExtraTrees Regressor

# %%
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
model(etr,train_X,train_Y,test_X,test_Y,"feat")

# %% [markdown]
# # Decision Tree Regressor

# %%
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
model(dtr,train_X,train_Y,test_X,test_Y,"feat")

# %% [markdown]
# # Random Forest Regressor

# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model(rf,train_X,train_Y,test_X,test_Y,"feat")

# %% [markdown]
# # Gradient Boosting Regressor

# %%
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
model(gbr,train_X,train_Y,test_X,test_Y,"feat")

# %% [markdown]
# # XGBoost Regressor

# %%
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
xgr =XGBRegressor()
model(xgr,train_X,train_Y,test_X,test_Y,"feat")


# %%



# %%



