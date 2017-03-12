#ML
bluescore = testbench for 

Most time here >> Model - a target function . set of parameters.
different for different Tasks.
It should represent the task correctly.


Different learning algos - grad descent.
+adagrad, adam 

SVM dos- has Large Margin Classifier
they define the marginal line


Training Datas=
Categorical variables = {1...k}
WildernessArea1 <- (one-not) representation =  It is observed that the training is less accurate ( 977 -> 927 / 1500 )
Class Distribution
print(dataset.groupby('Soil_Type33').size())  #group by and output the no of each type k
^To identify class bias in our database	
>> We removed that Soil_Type15 coz it will shift our confidence about the 


sklearn.linear_model.LogisticRegression(
	C=1e5 # 10^5
penalty  ='l2', # Regularisations - l2 <- andrew ng uses it
solver   = newtons method, liblinear, etc # try diff methods and see which one 
max_iter =100 , #
)

data = array(dataset) # from numpy. Converts 'dataframe' of panda, into an n-dimensional array.
#sklearn can only operate on arrays.

PROBLEM STATEMENT-
Forest tree type pred


code - 
train.drop('SalePrice',axis=1,inplace=True) <- modify same variable instead of train = train.drop
logreg= linear_model.LogisticRegression(penalty='l1',C=1e5,max_iter=10) # developing our model.
# logistic model ka object ban gaya

logreg.fit(X,Y) # fit means train data by input array X and supervised learning data output
# so fit means we apply h=Theta0 + Theta1.X


predic.csv is to be passed to Make a submission on Kaggle/

10 iter = 38%
100 iter = 55%

getting a 1.0000
use Transformations Polynomial regression

Skewing the J function 
-> sklearn preprocessing does this for you.
StandardScaler(). finds the mean value and mean - Kstd -> mean + Kstd


/*
choosing the Model

Lot of Categoricals = Random Forest
Less No of vars & continous vars = LogisticRegression
Lot of continous = Linear Regression

*/

size = 10 # first 10 columns are my continous variables. Later ones are Categoricals
X_temp = preprocessing.StandardScaler().fit_transform(X_train[:,0:size]) #only the continous
X_val_temp = preprocessing.StandardScaler().fit_transform(X_val[:,0:size]) #only the continous
#Concatenate non-categorical data
X_con = numpy.concatenate((X_temp,X_train[:,size:]),axis=1); # The Scaled Columns are joined to the Categorical
X_val_con = numpy.concatenate((X_val_temp,X_val[:,size:]),axis=1); # The Scaled Columns are joined to the Categorical



COool function = 
get_dummies(all_data) will split given column into K columns according to what values it sees.
To split different int64 values, use astype('category')



Development Set = Used for testing our constants Alpha's & C's 	



/***
Regularisation =  prevents overfitting ( too much accurate on training is not good.)
l1 loss = LASSO = 
J = [] + Lamba(Theta1 + Theta0..)
l2 loss = RIDGE
J = [] + Lamba(Theta1^2 + Theta0^2..)
/^above lamba aka alpha  


HW -
Read SKLEARN Documentations
Kaggle Links - submit the problems
try svm models
ML's learning curve IS Steep, Don't leave it , keep it up.