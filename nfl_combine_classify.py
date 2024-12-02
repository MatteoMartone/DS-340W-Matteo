

from nfl_combine_regressor import nflCombineRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class nflCombineClassify(nflCombineRegressor):
    
    def __init__(self,path):
        super().__init__()
        super().read_in(path) 
        super().cumulative_snaps()
        
    def snaps_to_binary(self):
        cols = ['40yd', 'Vertical', 'BP', 'Broad Jump', 'Shuttle', '3Cone']
    
        # Extract relevant features and drop NaN rows
        x_data_13_ = self.pd_2013[cols].dropna()
        x_data_14_ = self.pd_2014[cols].dropna()
        x_data_15_ = self.pd_2015[cols].dropna()
        x_data_16_ = self.pd_2016[cols].dropna()
        x_data_17_ = self.pd_2017[cols].dropna()

        y_data_13_nonan = self.snaps_cum_2013.loc[x_data_13_.index]
        y_data_14_nonan = self.snaps_cum_2014.loc[x_data_14_.index]
        y_data_15_nonan = self.snaps_cum_2015.loc[x_data_15_.index]
        y_data_16_nonan = self.snaps_cum_2015.loc[x_data_16_.index]
        y_data_17_nonan = self.snaps_cum_2017.loc[x_data_17_.index]

        # Convert target values to binary
        y_data_13_nonan = (y_data_13_nonan > 0).astype(int)
        y_data_14_nonan = (y_data_14_nonan > 0).astype(int)
        y_data_15_nonan = (y_data_15_nonan > 0).astype(int)
        y_data_16_nonan = (y_data_16_nonan > 0).astype(int)     
        y_data_17_nonan = (y_data_17_nonan > 0).astype(int)
        
        # Combine all data
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan, y_data_16_nonan, y_data_17_nonan])
        x = pd.concat([x_data_13_, x_data_14_, x_data_15_,x_data_16_, x_data_17_])
        
        x = x.apply(pd.to_numeric, errors='coerce').fillna(0)

        self.x_train_classify, self.X_test_classify, self.y_train_classify, self.y_test_classify = train_test_split(x, y, test_size=0.3, random_state=42)

    def model_test_classify(self):
        
        self.model1_classify = DecisionTreeClassifier(criterion='entropy', random_state=42)
        self.model2_classify = GradientBoostingClassifier(n_estimators=105,max_depth=4,tol=0.001, random_state=42)
        self.model3_classify = SVC(kernel='linear')
        self.model4_classify = GaussianNB()
        self.model5_classify = RandomForestClassifier(n_estimators=105,criterion='entropy',min_samples_leaf=4, random_state=42)
        self.model6_classify = LogisticRegression(max_iter=105, random_state=42)
        
        self.model1_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model2_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model3_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model4_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model5_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model6_classify.fit(self.x_train_classify,self.y_train_classify)

        
        y_pred1 = self.model1_classify.predict(self.X_test_classify)
        y_pred2 = self.model2_classify.predict(self.X_test_classify)
        y_pred3 = self.model3_classify.predict(self.X_test_classify)
        y_pred4 = self.model4_classify.predict(self.X_test_classify)
        y_pred5 = self.model5_classify.predict(self.X_test_classify)
        y_pred6 = self.model6_classify.predict(self.X_test_classify)

        print("DecisionTreeClassifier Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred1))
        print("GradientBoostingClassifier Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred2))
        print("SVC Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred3))
        print("GaussianNB Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred4))
        print("RandomForestClassifier Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred5))
        print("LogisticRegression Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred6))
        
    def plot_feature_importance_classify(self):
        imps = permutation_importance(self.model4_classify, self.X_test_classify, self.y_test_classify)
        #Calculate feature importance 
        feature_imp1 = pd.Series(self.model1_classify.feature_importances_,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp2 = pd.Series(self.model2_classify.feature_importances_,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp4 = pd.Series(imps.importances_mean,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp3 = pd.Series(self.model3_classify.coef_[0],index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp5 = pd.Series(self.model5_classify.feature_importances_,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp6 = pd.Series(self.model6_classify.coef_[0],index=self.X_test_classify.columns).sort_values(ascending=False)
        fig, axs = plt.subplots(2, 3)
        axs = axs.flatten()
    
        sns.barplot(ax=axs[0],x=feature_imp1,y=feature_imp1.index)
        sns.barplot(ax=axs[1],x=feature_imp2,y=feature_imp2.index)
        sns.barplot(ax=axs[2],x=feature_imp3,y=feature_imp3.index)
        sns.barplot(ax=axs[3],x=feature_imp4,y=feature_imp4.index)
        sns.barplot(ax=axs[4],x=feature_imp5,y=feature_imp5.index)
        sns.barplot(ax=axs[5],x=feature_imp6,y=feature_imp6.index)
        plt.xlabel('Feature Importance')
        axs[0].set_title('DecisionTreeClassifier')
        axs[1].set_title('GradientBoostingClassifier')
        axs[2].set_title('SVC')
        axs[3].set_title('GaussianNB')
        axs[4].set_title('RandomForestClassifier')
        axs[5].set_title('LogisticRegression')
        plt.draw()
        plt.show()
            
if __name__ == '__main__':
    classify = nflCombineClassify('')
    classify.snaps_to_binary()
    classify.model_test_classify()

    classify.plot_feature_importance_classify()




    