import numpy as np;
import os;
import re;
import sys;

from mpl_toolkits.mplot3d import Axes3D;
import matplotlib.pyplot as plt;
import pylab as pl;
import scipy.stats.mstats;
from sklearn.multiclass import OneVsRestClassifier;
from sklearn import svm;
from sklearn import lda;
from sklearn import qda;
from sklearn import tree;
from sklearn import neighbors;
from sklearn.naive_bayes import GaussianNB;

from sklearn.externals import joblib

def best_indices(times, tol):
	stimes = np.sort(times);
	astimes = np.argsort(times);
	i=0;
	while(stimes[i]/stimes[0] <= tol):
		i+=1;
	return astimes[0:i];

def svm_3d_vis(clf, X, t):
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	ax.scatter(X[:, 0], X[:, 1], X[:,2], c=t);
	ax.set_xlabel('M')
	ax.set_ylabel('N')
	ax.set_zlabel('K')
	ax.grid();
	plt.show();
	

train_path = sys.argv[1];
test_path = sys.argv[2];

print("Scanning files...");
X = [];
Y = [];
for fname in os.listdir(train_path):
	MNK = re.search(r"NT-float-([0-9]+)-([0-9]+)-([0-9]+).csv", fname);
	if MNK is not None:
		fl = open(train_path+fname,"rb");
		A = np.loadtxt(fl,delimiter=',');
		M = float(MNK.group(1));
		N = float(MNK.group(2));
		K = float(MNK.group(3));
		x = [(M-1632)/1000,(N-1632)/1000,(K-1632)/1000];
		X.append(x);
		Y.append(A[best_indices(A[:,0],1.02),1:]);

unique = [];

T = [None] * len(Y);
for idx,e in enumerate(Y):
	T[idx] = [];
	for y in e:
		ty = tuple(y);
		if ty in unique:
			T[idx].append(unique.index(ty));
		else:
			unique.append(ty);
			T[idx].append(len(unique) - 1);
X = np.array(X);
T = np.array(T);

print("Training classifier...");
clf = svm.SVC(kernel='rbf');
#clf = svm.NuSVC(kernel = 'linear', nu=0.5);
##clf = tree.DecisionTreeClassifier();
##clf = neighbors.KNeighborsClassifier();
##clf = GaussianNB();
clf = OneVsRestClassifier(clf);
clf.fit(X, T);
joblib.dump(clf, 'trained_model.pkl')

#svm_3d_vis(clf,X,t);

clf = joblib.load('trained_model.pkl')
ratios_predicted = [];
ratios_default = [];

print("Evaluating performance...");
default = np.array([ 2.,  8.,  8.,  8.,  4.,  1.,  6.,  1.,  1.,  8.,  8.]);
for fname in os.listdir(test_path):
	MNK = re.search(r"NT-float-([0-9]+)-([0-9]+)-([0-9]+).csv", fname);
	if MNK is not None:
		fl = open(test_path+fname,"rb");
		M = float(MNK.group(1));
		N = float(MNK.group(2));
		K = float(MNK.group(3));
		x = [(M-1632)/1000,(N-1632)/1000,(K-1632)/1000];
		[labels] = clf.predict(x);
		if len(labels) == 0:
			label = clf.label_binarizer_.classes_[np.argmax([e.decision_function(x) for e in clf.estimators_])];
		else:
			label = labels[0];
		y = np.array(unique[label]);
		A = np.loadtxt(fl,delimiter=',');
		idx_predicted = np.where(np.all(A[:,1:]==y,axis=1))[0];
		idx_default = np.where(np.all(A[:,1:]==default,axis=1))[0];
		opt = np.min(A[:,0]);
		ratios_default.append(opt/A[idx_default[0],0])
		ratios_predicted.append(opt/A[idx_predicted[0],0]);
		
gmean_predicted = scipy.stats.mstats.gmean(ratios_predicted);
gmean_default = scipy.stats.mstats.gmean(ratios_default);

np.set_printoptions(precision = 3);
#print(ratios_default);
#print(ratios_predicted);
print("GMeans - Predicted : %f, Default : %f"%(gmean_predicted,gmean_default));
