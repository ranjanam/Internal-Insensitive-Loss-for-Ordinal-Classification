import matplotlib.pyplot as plt
import binning
import sys
import csv
import numpy as np
from cvxopt import solvers
from cvxopt import matrix
import random


filename = "machine"
C=1
lam = 1000
reg = -1 * float(1/lam)
map_range = {}

#define 'b' based on class ranges
def calculate_b(map_range):
	b = {}
	for key in map_range.keys():
		range = map_range[key]
		b[key] = [range[0]+0.5, range[1]+0.5]
	return b

#define bin number or class label for sample label
def classLabel(value, map_range):
	value = float(value)
	key_set = map_range.keys()
	result=0
	key_set.sort()
	for key in key_set:
		if map_range[key][0]< value and map_range[key][1]>=value:
			result=key
			break	
	return result

#read data from file
def read_data(filename, map_range):
	data = []
	count = 0
	with open(filename,'r') as f:
		reader = csv.reader(f)
		for row in reader:
			data.append([float(x) for x in row])
			count+=1
	return count, data

#generate data for kFold
def generate (index, data) :
	data_t = []
	for i in xrange(len(index)):
		data_t.append(data[index[i]])
	return data_t


# expected_label (y)
# case 1: predicted variable lies in one of the ranges [y-1,y], [y,y+1], [y-1,y+1] which is selected randomly
def partial1 ( pre_y, ex_y):
	a = int(random.random() * 3)
	if a%3 == 0:
		return pre_y >= ex_y-1 and pre_y <= ex_y+1
	elif a%3 ==1:
		return pre_y >= ex_y and pre_y <= ex_y+1
	elif a%3 == 2: 
		return pre_y >= ex_y-1 and pre_y <= ex_y
	return False 

# expected_label (y)
# case 2: predicted variable lies in one of the ranges [y-1,y], [y,y+1], [y-1,y+1], 
# [y-2,y],[y,y+2],[y-2,y+2], [y-2,y+1], [y-1,y-2] which is selected randomly
def partial2 ( pre_y, ex_y):
	a = int(random.random() * 8)
	if a%8 == 0:
		return pre_y >= ex_y-1 and pre_y <= ex_y
	elif a%8 == 1:
		return pre_y >= ex_y and pre_y <= ex_y+1
	elif a%8 == 2: 
		return pre_y >= ex_y-1 and pre_y <= ex_y+1
	elif a%8 == 3:
		return pre_y >= ex_y-2 and pre_y <= ex_y
	elif a%8 == 4: 
		return pre_y >= ex_y and pre_y <= ex_y+2
	elif a%8 == 5:
		return pre_y >= ex_y-2 and pre_y <= ex_y+2
	elif a%8 == 6: 
		return pre_y >= ex_y-2 and pre_y <= ex_y+1
	elif a%8 == 7:
		return pre_y >= ex_y-1 and pre_y <= ex_y+2
	else :
		return 

#calculate MAE loss
def loss(y, rang):
	if y<= rang[1] and y >= rang[0]:
		return 0
	elif y > rang[1]:
		return y - rang[1]
	elif y < rang[0] : 
		return  - y + rang[0]
	return 0

#test data
def test(data, W, map_range):
	accuracy1=0
	accuracy2=0
	MAE1=0
	MAE2=0
	for d in data:
		expected_label = classLabel(d[-1],map_range)
		x = np.array(d[:-1])
		observed_label = classLabel(x.dot(W), map_range)
		l = loss(x.dot(W), map_range[expected_label])

		if partial1(observed_label, expected_label):
			accuracy1+=1
		else:
			MAE1 += l

		if partial2(observed_label, expected_label):
			accuracy2+=1
		else : 
			MAE2 += l

	return float(accuracy1)/len(data) , float(accuracy2)/len(data), float(MAE1)/len(data), float(MAE2)/len(data)


def compute(arg):
	map_range = binning.main(arg)
	
	data_length, full_data = read_data(filename, map_range)

	from sklearn import model_selection
	kf = model_selection.KFold(n_splits=5, shuffle=True)

	acc1=[]
	acc2=[]
	MAE1=[]
	MAE2=[]
	for train_index, test_index in kf.split(full_data):
		# training the sample
		data = generate(train_index, full_data)
		data_length = len(data)

		#cacluating variables
		b_map = calculate_b(map_range)
		A = [1.0]*2*data_length
		b = [0.0]
		G = np.identity(2*data_length)
		h = [float(C/data_length)]*2*data_length
		P_left = []
		P_right= []

		#calcuate P
		for d in data:
			label = classLabel(d[-1], map_range)
			rang = map_range[label]
			b_rang = b_map[label]
			list_left = [0]
			list_right = [0]
			
			for t in data:
				if rang[0] >= t[-1]:
					list_left.append(rang[0]-t[-1]-b_rang[0])
				if rang[1] <= t[-1]:
					list_right.append(rang[1]-t[-1]-b_rang[1])

			P_left.append(max(list_left))
			P_right.append(max(list_right))

		P = P_left  + P_right
		Q=[[0]*2*data_length]*2*data_length
		Q = np.array(Q)


		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
		
			for j, data_j in enumerate(data):
				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
				list_left = [0]
				list_right = [0]
				
				for t in data:
					if rang_i[0] >= t[-1]:
						list_left.append(t[-1]-rang_i[0])
					if rang_j[0] >= t[-1]:
						list_right.append(t[-1]-rang_j[0])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i][j] = x_i.dot(x_j) * max(list_left) * max(list_right) * reg

		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
			for j, data_j in enumerate(data):
				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
				list_right = [0]
				list_left = [0]
				
				for t in data:
					if rang_i[1] <= t[-1]:
						list_left.append(t[-1]-rang_i[1])
					if rang_j[1] <= t[-1]:
						list_right.append(t[-1]-rang_j[1])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i][data_length+j] = max(list_left) * max(list_right) * x_i.dot(x_j) * reg

		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
			
			for j, data_j in enumerate(data):
				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
				list_left = [0]
				list_right = [0]
				#y loop
				for t in data:
					if rang_i[0] >= t[-1]:
						list_left.append(t[-1]-rang_i[0])
					if rang_j[1] <= t[-1]:
						list_right.append(t[-1]-rang_j[1])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i+data_length][j] = max(list_left) * max(list_right) * x_i.dot(x_j) *reg

		for i, data_i in enumerate(data):
			label_i = classLabel(data_i[-1], map_range)
			rang_i = map_range[label_i]
			for j, data_j in enumerate(data):

				label_j = classLabel(data_j[-1], map_range)
				rang_j = map_range[label_j]
			
				list_left = [0]
				list_right = [0]
			
				for t in data:
					if rang_i[1] <= t[-1]:
						list_left.append(t[-1]-rang_i[1])
					if rang_j[0] >= t[-1]:
						list_right.append(t[-1]-rang_j[0])

				x_i = np.array(data_i[:-1])
				x_j = np.array(data_j[:-1])
				Q[i+data_length][j+data_length] = max(list_left) * max(list_right) * x_i.dot(x_j) * reg

		sol = solvers.qp(matrix(np.array(Q), tc='d'),
							matrix(np.array(P), tc='d'),
						 	matrix(np.array(G), tc='d'),
						 	matrix(np.array(h), tc='d'),
						 	matrix(np.array(A),(1,len(A)), tc='d'),
						 	matrix(np.array(b), tc='d')	
						)

		W=[]
		if sol['status'] == 'optimal':	
			# w generation
			Alpha = sol['x']

			for i, data_i in enumerate(data):
				label = classLabel(data_i[-1], map_range)
				rang = map_range[label]

				alpha_i = Alpha[i]
				alpha_i_star = Alpha[i+data_length]

				list_left = [0]
				list_right = [0]
				x_i = np.array(data_i[:-1])
			
				for t in data:
					if rang[0] >= t[-1]:
						list_left.append(t[-1]-rang[0])
					if rang[1] <= t[-1]:
						list_right.append(t[-1]-rang[1])

				temp1 = max(list_left)*alpha_i
				temp1 = x_i.dot(temp1)
				temp2 = max(list_right)*alpha_i_star
				temp2 = x_i.dot(temp2)

				if len(W)==0:
					W = np.add(temp1, temp2)
				else:
					temp1 = np.add(temp1, temp2)
					W = np.add(W, temp1)	

			W = W*reg

			#testing data
			data_test = generate(test_index, full_data)
			accuracy1, accuracy2, mae1, mae2 = test(data_test, W, map_range)
			acc1.append(accuracy1)
			acc2.append(accuracy2)
			MAE1.append(mae1)
			MAE2.append(mae2)
	
	print arg
	print "Accuracy : ", np.array(acc1).mean(), np.array(acc2).mean()
	print  "MAE : ",  np.array(MAE1).mean(), np.array(MAE2).mean()
	print "==============================================================="
	return np.array(MAE1).mean(), np.array(MAE2).mean()

if __name__ == '__main__':
	C=1
	bin_size = [10,50,100,150,200]
	case1=[]
	case2=[]
	for arg in bin_size:
		case1_mae, case2_mae = compute(arg)
		case1.append(case1_mae)
		case2.append(case2_mae)

	# plot
	fig = plt.figure()
	fig.suptitle('SVOR-IIL-IMC', fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)
	fig.subplots_adjust(top=0.85)
	ax.set_xlabel('sample size')
	ax.set_ylabel('MAE of unnormalized data')
	ax.plot(bin_size,case1,"bo")
	ax.plot(bin_size,case2,"ro")
	plt.show()
