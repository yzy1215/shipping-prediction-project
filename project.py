# Predicting the shipment punctuaity
# Import and load the data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# Inspecting the data
#note that column 0,3,4,5,6,9,10,11 are int64 type and column 1,2,7,8 are object type

# From inspection, the data does not have any missing value

#Splitting the data into train and test sets
def define_xy(data):
	column_name = list(data)
	y = data.iloc[:, 11]
	y.rename = ("Customer_rating")
	X = data.iloc[:, 0:11]
	return X, y

#Converting the non-numeric values into numeric values
def preprocessing(X,y):
	X=X.drop(columns=['ID'])
	le = LabelEncoder()
	for col in X:
		if X[col].dtypes =='object':
			X[col]=le.fit_transform(X[col])
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=42)
	scaler = MinMaxScaler()
	normalizedX_train = pd.DataFrame(scaler.fit_transform(X_train), columns= list(X_train))
	normalizedX_test = pd.DataFrame(scaler.transform(X_test), columns= list(X_test))
	Y_train = pd.DataFrame(y_train)
	Y_test = pd.DataFrame(y_test)

	return normalizedX_train, normalizedX_test, Y_train, Y_test

def pearson_corr(xTrain, yTrain):
	#print(xTrain)
	
	n_feature = xTrain.shape[1]

	w, h = n_feature + 1, n_feature + 1;
	corr = [[0 for x in range(w)] for y in range(h)]

	for i in range(n_feature):
	    for j in range(n_feature):
	        p_corr, _ = pearsonr(xTrain.iloc[:, i], xTrain.iloc[:, j])
	        corr[i][j] = p_corr

	for i in range(n_feature):
		
	    p_corr, _ = pearsonr(xTrain.iloc[:, i], yTrain['Reached.on.Time_Y.N'])
	    #print(p_corr)
	    corr[i][n_feature] = p_corr
	    corr[n_feature][i] = p_corr

	corr[n_feature][n_feature] = 1

	return corr


def select_features(df, corr):
    len_corr = len(corr)
    for i in range(len_corr):
        for j in range(len_corr):
            corr[i][j] = abs(corr[i][j])
            corr[j][i] = abs(corr[j][i])
    corr = np.array(corr)

    # filter out the most uncorrlated features, and keep the 5 most correlated features
    sorted_target_corr = np.sort(corr[10])[::-1]
    sorted_target_arg = np.argsort(corr[10])[::-1]
    top_5_feature_target = sorted_target_arg[1:6]

    # find the closely related features and store in close_feature

    close_feature = list()
    corrsub = corr[0:-1, 0:-1]
    for i in range(len_corr - 1):
        for j in range(i + 1, len_corr - 1):
            if abs(corrsub[i][j]) > 0.7:				## define closely realted as correlation over 0.7
                close_feature.append([i, j])

    remove_ind = list()									## find the features we want to remove
    if len(close_feature)!= 0:								## if there are closely related features in top_5_featues, remove one of them
	    for ele in close_feature:		
	        if ele[0] in top_5_feature_target:				
	            if ele[1] in top_5_feature_target:
	                top_feature = np.delete(top_5_feature_target, np.where(top_5_feature_target == ele[0]))	


	    
	    
	    for i in range(len_corr - 1):
	        if i not in top_feature:
	            remove_ind.append(i)
	    
    else:												## if there is no close_features
    	for i in range(len_corr - 1):					## simply remove the fetures not in top_5_features
	        if i not in top_5_feature_target:
	            remove_ind.append(i)
	            #print(remove_ind)
	
    df = df.drop(df.columns[remove_ind], axis = 1)
			
    return df


def main():
	data=pd.read_csv("Train.csv",header =0)

	## preprocessing the data
	X, y = define_xy(data)
	normalizedX_train, normalizedX_test, Y_train, Y_test = preprocessing(X,y)
	corr_train = pearson_corr(normalizedX_train,Y_train)
	fig = plt.figure(figsize=(10, 8))
	ax1 = fig.add_subplot(1, 1, 1)
	sns.heatmap(corr_train, ax=ax1, vmin=-1, vmax=1, center=0,annot=True)
	df_Train = select_features(normalizedX_train, corr_train)
	df_Test = select_features(normalizedX_test, corr_train)

	## first method -- logistic regression
	logreg = LogisticRegression(max_iter=100, tol=0.0001)
	logreg.fit(df_Train, np.ravel(Y_train))
	y_pred = logreg.predict(df_Test)
	print("Accuracy of logistic regression classifier before tuning: ",logreg.score(df_Test, Y_test))
	tol = [0.01,0.001,0.0001]
	max_iter = [100,150,200]
	param_grid = {"tol":tol,"max_iter": max_iter}
	grid_model = GridSearchCV(estimator= logreg, param_grid=param_grid, cv=5)
	grid_model_result = grid_model.fit(normalizedX_train,np.ravel(Y_train))
	best_params = grid_model_result.best_params_
	print("   best_params_", best_params)
	logreg2 = LogisticRegression(max_iter=100, tol=0.01)
	logreg2.fit(df_Train, np.ravel(Y_train))
	y_pred = logreg2.predict(df_Test)
	print("Accuracy of logistic regression classifier after tuning: ",logreg2.score(df_Test, Y_test))


	## Second method -- K nearest neighbors
	model = KNeighborsClassifier(n_neighbors = 3)
	model.fit(normalizedX_train, np.ravel(Y_train))
	y_pred = model.predict(normalizedX_test)
	print("Accuracy of KNN classifier before hyperparamenter tuning:", metrics.accuracy_score(Y_test, y_pred))
	grid_params = {'n_neighbors':[200,300,400,500], 'metric':['euclidean','manhattan']}
	gs = GridSearchCV(KNeighborsClassifier(), grid_params)
	gs_results = gs.fit(normalizedX_train, np.ravel(Y_train))
	best_n_neighbors = gs_results.best_params_['n_neighbors']
	best_metric = gs_results.best_params_['metric']
	print("    best nunber of neighbors:", best_n_neighbors, "best metric:", best_metric)

	## Third method -- Decision tree classifier
	classifier = DecisionTreeClassifier()
	classifier.fit(normalizedX_train, np.ravel(Y_train))
	y_pred = classifier.predict(normalizedX_test)
	print("Accuracy of Decision classifier:", metrics.accuracy_score(Y_test, y_pred))

	## hyper parameter tuning for KNN classifier
	model2 = KNeighborsClassifier(n_neighbors= best_n_neighbors, metric = best_metric)
	model2.fit(normalizedX_train, np.ravel(Y_train))
	y_pred = model2.predict(normalizedX_test)
	print("Accuracy of KNN classifier after hyperparamenter tuning:", metrics.accuracy_score(Y_test, y_pred))

	plt.show()




if __name__ == "__main__":
    main()
