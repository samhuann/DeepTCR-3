import pandas as pd
from DeepTCR3.DeepTCR3 import DeepTCR3_SS
import numpy as np
import pickle

df = pd.read_csv('../../../Data/10x_Data/Data_Regression.csv')
antigen = 'A0201_GILGFVFTL_Flu-MP_Influenza'

DTCRS = DeepTCR3_SS('reg_flu',device=2)
#Get alpha/beta sequences
alpha = np.asarray(df['alpha'].tolist())
beta = np.asarray(df['beta'].tolist())
i = np.where(df.columns==antigen)[0][0]
sel = df.iloc[:, i]
Y = np.log2(np.asarray(sel.tolist()) + 1)
DTCRS.Load_Data(alpha_sequences=alpha, beta_sequences=beta, Y=Y)
folds = 5
seeds = np.array(range(folds))
graph_seed = 0
DTCRS.K_Fold_CrossVal(split_by_sample=False, folds=folds,seeds=seeds,graph_seed=graph_seed)

with open('flu_preds.pkl','wb') as f:
    pickle.dump([antigen,np.squeeze(DTCRS.predicted),Y],f,protocol=4)

