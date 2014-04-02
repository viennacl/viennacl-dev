import numpy as np;
import os;
import re;
import sys;
import random;

from mpl_toolkits.mplot3d import Axes3D;
import matplotlib.pyplot as plt;
import pylab as pl;
import scipy.stats;
from sklearn import svm;
from sklearn import neighbors;
from sklearn import preprocessing;
from sklearn.externals import joblib;
from sklearn.metrics import accuracy_score;

def best_indices(times, tol):
	stimes = np.sort(times);
	astimes = np.argsort(times);
	i=0;
	while(stimes[i]/stimes[0] <= tol):
		i+=1;
	return astimes[0:i];

def svm_3d_vis(X, t):
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	M = X[:, 0];
	N = X[:, 1];
	K = X[:, 2];
	ax.scatter(M,N,K, c=t);
	ax.set_xlabel('M')
	ax.set_ylabel('N')
	ax.set_zlabel('K')
	ax.grid();
	plt.show();
	

path = sys.argv[1];

print("Scanning files...");
X = [];
Y = [];
O = [];
files = os.listdir(path);
random.seed(0);
random.shuffle(files);
for fname in files:
	MNK = re.search(r"NT-float-([0-9]+)-([0-9]+)-([0-9]+).csv", fname);
	if MNK is not None:
		fl = open(path+fname,"rb");
		A = np.loadtxt(fl,delimiter=',');
		M = float(MNK.group(1));
		N = float(MNK.group(2));
		K = float(MNK.group(3));
		x = [M,N,K];
		X.append(x);
		Y.append(map(tuple,A[best_indices(A[:,0],1.07),1:]));

scaler = preprocessing.StandardScaler().fit(X);
X = scaler.transform(X)
unique = [];
counts = [];
for e in Y:
	for y in e:
		if y in unique:
			counts[unique.index(y)]+=1;
		else:
			unique.append(y);
			counts.append(1);
t=[];
for e in Y:
	x = [];
	for y in e:
		x.append(counts[unique.index(y)]);
	t.append(unique.index(e[np.argmax(x)]));
tmp=list(set(t));
unique = [unique[k] for k in tmp];
t = [tmp.index(k) for k in t];

X = np.array(X);
t = np.array(t);
ratio = 0.5;

X_train = X[1:X.shape[0]*ratio,:];
t_train = t[1:X.shape[0]*ratio];

X_test = X[X.shape[0]*ratio+1:,:];
t_test = t[X.shape[0]*ratio+1:];

#svm_3d_vis(X,t);

print("Training classifier...");
clf = svm.SVC(kernel='rbf');
#clf = svm.NuSVC();
#clf = tree.DecisionTreeClassifier();
#clf = neighbors.KNeighborsClassifier(1);
##clf = GaussianNB();
#clf = svm.LinearSVC();
clf.fit(X_train, t_train);
joblib.dump(clf, 'trained_model.pkl')
clf = joblib.load('trained_model.pkl')

print("Evaluating performance...");
print("Accuracy score : %f"%(accuracy_score(t_test, clf.predict(X_test))));

default = np.array([ 2.,  8.,  8.,  8.,  4.,  1.,  6.,  1.,  1.,  8.,  8.]);
ratios_predicted = [];
ratios_default = [];

for x in X_test:
	[tM, tN, tK] = scaler.inverse_transform(x);
	[M,N,K] = map(int,map(round,[tM,tN,tK]));
	fl = open(path+"NT-float-"+`M`+"-"+`N`+"-"+`K`+".csv","rb");
	label = clf.predict(x);
	y = np.array(unique[label]);
	A = np.loadtxt(fl,delimiter=',');
	idx_predicted = np.where(np.all(A[:,1:]==y,axis=1))[0];
	idx_default = np.where(np.all(A[:,1:]==default,axis=1))[0];
	opt = np.min(A[:,0]);
	ratios_default.append(opt/A[idx_default[0],0])
	ratios_predicted.append(opt/A[idx_predicted[0],0]);
		

np.set_printoptions(precision = 2);
print("GMeans - Predicted : %f, Default : %f"%(scipy.stats.mstats.gmean(ratios_predicted), scipy.stats.mstats.gmean(ratios_default)));
print("Median - Predicted : %f, Default : %f"%(np.ma.median(ratios_predicted) ,np.ma.median(ratios_default)));
print("AMean - Predicted : %f, Default : %f"%(np.ma.mean(ratios_predicted),np.ma.mean(ratios_default)));
