import pandas as pd
from random import random
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from math import exp, log, sqrt
from sklearn.feature_extraction.text import CountVectorizer
from torch.autograd import Variable
import torch
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


data_og = pd.read_csv('~/Desktop/Thesis/Data/gender.csv', index_col = 0)
##Copied from R, cases in which the Solicitor General Appears
general_cases = ['2015_68', '2012_0',  '2012_42', '2015_14', '2011_66', '2012_53', '2015_49', '2014_50', '2013_14', '2015_30',\
				 '2011_12', '2014_11', '2013_12', '2014_42', '2013_61', '2012_65', '2011_13', '2013_31', '2011_54', '2014_27',\
				 '2015_0',  '2012_20', '2012_30', '2011_29', '2011_15', '2011_46', '2012_21', '2013_17', '2014_58', '2011_18',\
				 '2013_49', '2014_36', '2011_5',  '2012_74', '2014_4',  '2013_24', '2015_31']
data = data_og[data_og['case_id'].isin(general_cases)]
print(len(data))

text = np.array(data['text'])
tag = np.array(data['gender'])




dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

vectorizer = CountVectorizer(analyzer='word', token_pattern = r'(?u)\b\w+\b')

w2 = vectorizer.fit_transform(text)

words = vectorizer.fit_transform(text).toarray()
# print(words.shape)
# print(words)

# # 0 == male, 1 == female
tags = [int(label == 'F') for label in tag]

tags = np.array(tags)
# print(tags)

trainProp = 0.70
valProp = 0.15

trainNum = int(len(text) * trainProp)
valNum = trainNum + int(len(text) * valProp)

idx = np.random.RandomState(seed=131).permutation(range(len(text)))
trainIdx = idx[:trainNum]
testIdx = idx[trainNum:valNum]
validIdx = idx[valNum:len(text)]

train_xLR = words[trainIdx]
train_yLR = tags[trainIdx]
train_x = text[trainIdx]


test_xLR = words[testIdx]
test_yLR = tags[testIdx]

valid_xLR = words[validIdx]
valid_yLR = tags[validIdx]

wordSet = vectorizer.get_feature_names()



####################
#Naive Bayes
####################
# m = np.arange(0.0, 1.0, 0.002) #With this, ideal is 0.0
m = np.arange(0.0, .1, 0.00002) #Still 0.0
nb_acc = []
for alpha in m:
	print(alpha)
	mnb = MultinomialNB(alpha = alpha, fit_prior = True)
	mnb.fit(train_xLR, train_yLR)
	y_valid_pred = mnb.predict(valid_xLR)
	nb_acc.append(metrics.accuracy_score(valid_yLR, y_valid_pred))



print("NB accuracy pred:")
maxes = nb_acc.index(max(nb_acc))
print(nb_acc[maxes])
ideal_alpha = 0.0#m[maxes]
print("NB ideal alpha: ")
print(ideal_alpha)
print("Aiming for")
print('0.82')
print('result')

mnb = MultinomialNB(alpha = ideal_alpha, fit_prior = True)
mnb.fit(train_xLR, train_yLR)
y_test_pred = mnb.predict(test_xLR)
print(metrics.accuracy_score(test_yLR, y_test_pred))


####################
#Logistic Regresssion
####################

x_train = Variable(torch.from_numpy(train_xLR), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(train_yLR), requires_grad=False).type(dtype_long)

x_valid = Variable(torch.from_numpy(valid_xLR), requires_grad=False).type(dtype_float)
y_validclasses = Variable(torch.from_numpy(valid_yLR), requires_grad=False).type(dtype_long)

x_test = Variable(torch.from_numpy(test_xLR), requires_grad=False).type(dtype_float)

dim_x = words.shape[1]
dim_out = 2

model_logreg = torch.nn.Sequential(
	torch.nn.Linear(dim_x, dim_out)
)

loss_train = []
loss_valid = []

learning_rate = 1e-5
N = 1000
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_logreg.parameters(), lr=learning_rate)

for t in range(N):
	if t % 100 == 0:
		print(t)
	y_pred = model_logreg(x_train)
	loss = loss_fn(y_pred, y_classes)
	loss_train.append(loss.data.numpy().reshape((1,))[0]/len(y_classes))

	y_validpred = model_logreg(x_valid)
	y_validtry = y_validpred.data.numpy()
	# print('accuracy on validation set: ', np.mean(np.argmax(y_validtry, 1) == valid_yLR))
	loss_v = loss_fn(y_validpred, y_validclasses)
	loss_valid.append(loss_v.data.numpy().reshape((1,))[0]/len(y_validclasses))

	model_logreg.zero_grad()
	loss.backward()
	optimizer.step()

plt.plot(range(N), loss_valid, color='b', label='validation')
plt.plot(range(N), loss_train, color='r', label='training')
plt.legend(loc='upper left')
plt.title("Gender: Learning curve for training and validation sets")
plt.xlabel("Number of iterations (N)")
plt.ylabel("Cross entropy loss")
plt.show()

y_validpred = model_logreg(x_valid)
y_validtry = y_validpred.data.numpy()
print('accuracy on validation set: ', np.mean(np.argmax(y_validtry, 1) == valid_yLR))
	
y_testpred = model_logreg(x_test).data.numpy()

# results on test set
print('accuracy on testing set: ', np.mean(np.argmax(y_testpred, 1) == test_yLR))

# weights (fn) from word_i to 'fake' (z0)
weight_f = model_logreg[0].weight[0,:]

# weights (rn) from word_i to 'real' (z1)
weight_r = model_logreg[0].weight[1,:]

# top 10 presence predict real (FemP)
# for each word exp(weight_r) / exp(weight_r)+ exp(weight_f)

changeDict1 = {}

for i in range(len(wordSet)):
	changeDict1[wordSet[i]] = exp(-weight_r[i]) / (exp(-weight_r[i]) + exp(-weight_f[i]))

FemP = sorted(changeDict1, key=changeDict1.get, reverse=False)[:10]
FemA = sorted(changeDict1, key=changeDict1.get, reverse=True)[:10]

for word in FemA:
	print('FemA', word, changeDict1[word])
for word in FemP:
	print('FemP', word, changeDict1[word])


