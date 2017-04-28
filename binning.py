import matplotlib.pyplot as plt
import numpy as np
from numpy import array , dot
import sys
import csv

map_range={}
filepath="machine"
destDir ="./"

#segregate data
def generate (index, data) :
	data_t = []
	for i in xrange(len(index)):
		data_t.append(data[index[i]])
	return data_t

#write data to file
def write_file(handle, data):
	for i in xrange(len(data)):
		handle.write(",".join(data[i])+"\n")

#define ranges for each class
def form_ranges(Y, k):
	x=0
	classLabel=1
	while x < 1000:
		yl = x
		x+=int(k)
		yr = x
		map_range[classLabel]=[yl,yr]
		classLabel+=1

#define classlabels for each label in sample data
def classLabel(value):
	value = float(value)
	key_set = map_range.keys()
	result=0
	key_set.sort()
	for key in key_set:
		if map_range[key][0]< value and map_range[key][1]>=value:
			result=key
			break
	return str(result)

#generate ordinal data
def ordinal_data(data):
	for x in xrange(len(data)):
		data[x][-1] = classLabel(data[x][-1])
	return data

#segregate to in bins
def binning(k):
	data=[]
	with open(filepath,'r') as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row)

	#number of classes
	Y = np.ceil(1.0 * len(data) / int(k))
	form_ranges(Y, k)
	data = ordinal_data(data)

	from sklearn import model_selection
	kf = model_selection.KFold(n_splits=5, shuffle=True)

	i=1
	for train_index, test_index in kf.split(data):
		data_train = generate(train_index, data)
		data_test = generate(test_index, data)
		
		# write train and test data in a file
		filename = "train_data_"+str(i)+".csv"
		handle = open(destDir+filename, "w+")
		write_file(handle, data_train)
		handle.close()
		
		filename = "test_data_"+str(i)+".csv"
		handle = open(destDir+filename, "w+")
		write_file(handle, data_test)
		handle.close()
		i+=1
	return map_range

def main(size):
	return binning(size)
	