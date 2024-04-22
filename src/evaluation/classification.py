# %%
import os
import sys
import numpy as np
import json
from collections import defaultdict
from clf_tools import get_datafiles
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dir = '../../data'
nprints_dir = f"{data_dir}/finetune_nprints"
output_dir = 'results.json'
datafiles = get_datafiles(nprints_dir)

results = defaultdict(dict)
for label, Y in tqdm(datafiles['labels'].items()):
	X_train, X_test, y_train, y_test = train_test_split(datafiles['data'], Y, test_size=0.2, random_state=42)
	models = {
		'rf': RandomForestClassifier(n_estimators=100, random_state=42),
		'svc': SVC(kernel='linear', C=1.0, random_state=42),
		'xgb': XGBClassifier(objective='multi:softmax', num_class=len(np.unique(Y)), use_label_encoder=False, seed=42)
	}
	for name, model in tqdm(models.items()):
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		results[label][name] = acc

with open(output_dir, 'w') as f:
	json.dump(results, f)
	