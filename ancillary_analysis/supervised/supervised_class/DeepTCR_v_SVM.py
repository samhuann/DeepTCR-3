"""
Supplementary Figure 17
"""

"""This script is used to benchmark DeepTCR3's Sequence Classifier
against an SVM and RF where the inputs for those latter machine learning
algorithms are the outputs of a K-mer search"""

import numpy as np
import pandas as pd
from DeepTCR3.DeepTCR3 import DeepTCR3_SS
from NN_Assessment_utils import *
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,roc_curve
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


DTCRS = DeepTCR3_SS('Sequence_C')
DTCRS.Get_Data(directory='../../../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)
kmer_features = kmer_search(DTCRS.beta_sequences)
clf_svm = SVC(probability=True)
clf_rf = RandomForestClassifier(n_estimators=100)

y_test_list = []
y_pred_list_dtcr = []
y_pred_list_svm = []
y_pred_list_rf = []

for i in range(100):
    DTCRS.Get_Train_Valid_Test()
    DTCRS.Train(use_only_seq=True, num_fc_layers=0, units_fc=256)
    y_pred_list_dtcr.append(DTCRS.y_pred)

    #Kmer
    clf_svm.fit(kmer_features[DTCRS.train[6]],np.argmax(DTCRS.train[-1],-1))
    svm_pred = clf_svm.predict_proba(kmer_features[DTCRS.test[6]])
    y_pred_list_svm.append(svm_pred)

    #RF
    clf_rf.fit(kmer_features[DTCRS.train[6]],np.argmax(DTCRS.train[-1],-1))
    rf_pred = clf_rf.predict_proba(kmer_features[DTCRS.test[6]])
    y_pred_list_rf.append(rf_pred)

    y_test_list.append(DTCRS.test[-1])


auc = []
method = []
antigen = []
for y_test,y_pred_svm,y_pred_dtcr,y_pred_rf in zip(y_test_list,y_pred_list_svm,y_pred_list_dtcr,y_pred_list_rf):
    for ii,t in enumerate(DTCRS.lb.classes_,0):
        auc.append(roc_auc_score(y_test[:,ii],y_pred_dtcr[:,ii]))
        method.append('DeepTCR3')
        antigen.append(t)

        auc.append(roc_auc_score(y_test[:,ii],y_pred_svm[:,ii]))
        method.append('SVM')
        antigen.append(t)

        auc.append(roc_auc_score(y_test[:,ii],y_pred_rf[:,ii]))
        method.append('RF')
        antigen.append(t)

df = pd.DataFrame()
df['Method'] = method
df['AUC'] = auc
df['Antigen']=antigen
df.to_csv('DeepTCR3_v_SVM_v_RF.csv',index=False)
sns.violinplot(data=df,x='Antigen',y='AUC',hue='Method')
sns.violinplot(data=df,x='Method',y='AUC',order=['DeepTCR3','RF','SVM'])
sns.boxenplot(data=df,x='Method',y='AUC',order=['DeepTCR3','RF','SVM'])


from scipy.stats import ttest_ind as ttest
idx_1 = df['Method']=='SVM'
idx_2 = df['Method']=='RF'
t,p_val = ttest(df[idx_1]['AUC'],df[idx_2]['AUC'])

avg_vals = []
methods = ['DeepTCR3','RF','SVM']
for m in methods:
    avg_vals.append(np.mean(df[df['Method'] == m]['AUC']))



