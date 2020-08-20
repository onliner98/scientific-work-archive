import os
import random
import numpy as np

def train_val_test_split(arr, val_split=0.2, test_split=0.2):
	random.shuffle(arr)

	val_idx = int(round(val_split*3400))
	test_idx = int(round(test_split*3400))
	arr = np.array(arr)

	val = arr[0:val_idx]
	test = arr[val_idx:val_idx+test_idx]
	train = arr[val_idx+test_idx:3400] 
	return train, val, test
	


def move_sample(src_dir, dest_dir, sample):
	src_path = os.path.join(src_dir, sample)
	dest_path = os.path.join(dest_dir, sample)
	os.rename(src_path, dest_path)


def split_class(src_path, dest_path, class_name):
	test_path = os.path.join(dest_path, 'test', class_name)
	train_path = os.path.join(dest_path, 'train', class_name)
	val_path = os.path.join(dest_path, 'val', class_name)
	class_path = os.path.join(src_path, class_name)

	os.makedirs(test_path, exist_ok=True)
	os.makedirs(train_path, exist_ok=True)
	os.makedirs(val_path, exist_ok=True)

	_,_,samples = next(os.walk(class_path))
	print(samples)

	train, val, test = train_val_test_split(samples)

	for sample in train:
		move_sample(class_path, train_path, sample)
	for sample in val:
		move_sample(class_path, val_path, sample)
	for sample in test:
		move_sample(class_path, test_path, sample)


def split_dataset(src_path, dest_path, classes):
	print('classes:', classes)

	os.makedirs(os.path.join(dest_path), exist_ok=True)
	os.makedirs(os.path.join(dest_path, 'test'), exist_ok=True)
	os.makedirs(os.path.join(dest_path, 'train'), exist_ok=True)
	os.makedirs(os.path.join(dest_path, 'val'), exist_ok=True)

	for class_name in classes:
		split_class(src_path, dest_path, class_name)



split_dataset('dataset_src', 'dataset_dest', ['drill', 'hammer', 'pliers', 'saw', 'screwdriver', 'wrench'])