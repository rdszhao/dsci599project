import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def get_label(file):
	labels = {
		'encryption': file.split('-')[-3],
		'resolution': file.split('.')[-4],
		'transport': file.split('.')[-2] 
	}
	return labels

def process_nprint(filepath):
	df = pd.read_csv(filepath)
	num_packet = df.shape[0]
	if num_packet != 0:
		substrings = ['ipv4_src', 'ipv4_dst', 'ipv6_src', 'ipv6_dst','src_ip']
		cols_to_drop = [col for col in df.columns if any(substring in col for substring in substrings)]
		df = df.drop(columns=cols_to_drop)
	width, height = df.shape[1], df.shape[0]
	padded_height = 1024
	np_img = np.full((padded_height, width), -1)
	np_df = np.array(df.apply(lambda x: x.apply(np.array)).to_numpy().tolist())
	np_img[:height, :] = np_df
	arr = np_img.flatten()
	return arr

def get_datafiles(nprints_dir):
	print(f"reading nprints from {nprints_dir}...")
	files = [file for file in os.listdir(nprints_dir) if '.nprint' in file]
	datafiles = np.vstack([process_nprint(f"{nprints_dir}/{file}") for file in tqdm(files)])
	labels = [get_label(file) for file in files]
	labelmap_raw = {key: np.array([d[key] for d in labels]) for key in labels[0]}
	labelmap_encoded = {key: LabelEncoder().fit_transform(value) for key, value in labelmap_raw.items()}
	datamap = {'data': datafiles, 'labels': labelmap_encoded}
	return datamap