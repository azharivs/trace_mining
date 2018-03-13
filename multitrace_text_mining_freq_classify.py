#!/usr/bin/python3.5
import pickle
import os.path
from collections import Counter
import babeltrace
import sys
import statistics
import numpy as np
#from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt1
import time
# Though the following import is not directly being used, it is required
# for 3D projection to work
#from mpl_toolkits.mplot3d import Axes3D

#plt.ion()
#colors = {0: '#000000', 1: '#15b01a', 7: '#0343df', 10: '#9a0eea', 12: '#e50000', 30: '#ff796c', 31: '#ffff14', 32: '#f97306', 48: '#00ffff', 49: '#01ff07', 52: '#75bbfd'}

# a trace collection holds one or more traces
col = babeltrace.TraceCollection()

# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)

duration = []
items = []
corpus=[]
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu', 'addons_vcpu_enter_guest']

trace_name = sys.argv[1].split('.')[0]
with open(sys.argv[1],'r') as f:
    trace_file_names = f.readlines()

trace_list=[]
for s in trace_file_names:
    if s.find('#') == -1: continue #skip tracefile starting with # for training and only use for test set 
    filename = s.splitlines()[0].split(':')[1]
    tid = int(s.splitlines()[0].split(':')[2])
    if filename == "": break
    filename = filename.replace("#","")
    trace_list.append(filename)    
    
    corpus_filename = filename.split('kernel')[0]+'vm.pkl'
    if os.path.exists(corpus_filename) == False: #no corpus file already exists so need to parse the trace
        trace_handle = col.add_trace(filename,'ctf')
        if trace_handle is None:
            raise RunTimeError('Cannot add trace')
        
        print('----------------- STARTING EVENT PROCESSING: ',filename)
        trace=''
        index = 0
       # iterate on events
        for event in col.events:
            if event['tid'] == tid:
                ts = event.timestamp
                if index == 1: 
                    start = ts
                #if index >= 100: 
                #    break
                if not (event.name in stopwords): index = index + 1
                trace = trace + ' ' + event.name.replace('_x86_','_')
                if 'exit_reason' in event: trace = trace + '_' + str(event['exit_reason']) 
                if 'vector' in event: trace = trace + '_' + str(event['vector'])
                if 'vec' in event: trace = trace + '_' + str(event['vec'])
                if 'irq' in event: trace = trace + '_' + str(event['irq'])
        dur = float(ts - start)/1000000
        col.remove_trace(trace_handle)
    else: #there is a corpus file for this trace so read from it
        print('================= READING FROM CORPUS FILE: ',corpus_filename)        
        f = open(corpus_filename,'rb')
        dur = pickle.load(f)
        index = pickle.load(f)
        trace = pickle.load(f)
        f.close()
        
    duration.append(dur)
    items.append(index)    
    corpus.append(trace)

print('Loading previously trained model from pickle file ............')
f = open(trace_name+'.train','rb')
X = pickle.load(f)
vectorizer = pickle.load(f)
tf_transformer = pickle.load(f)
tfidf_transformer = pickle.load(f)
labels = pickle.load(f)
f.close()

X_train_tfidf = tfidf_transformer.fit_transform(X)

print('===========MultinomialNB===========')
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)


print('===========SGDClassifier===========')
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None)
clf.fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)


print('==========SVC(decision_function_shape=ovo)============')
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)

print('==========SVC(decision_function_shape=ovr)============')
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)

print('==========SVC(kernel=linear, C=1.0)============')
svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)

print('===========LinearSVC(C=1.0)===========')
svm.LinearSVC(C=1.0)
clf.fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)

print('===========SVC(kernel=rbf, gamma=0.7, C=1.0)===========')
svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
clf.fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)

print('===========SVC(kernel=poly, degree=3, C=1.0)===========')
svm.SVC(kernel='poly', degree=3, C=1.0)
clf.fit(X_train_tfidf, labels)
X_new_counts = vectorizer.transform(corpus)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for p,t in zip(predicted, trace_list):
    print(p,':',t)


