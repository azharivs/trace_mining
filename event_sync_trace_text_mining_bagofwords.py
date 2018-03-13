#!/usr/bin/python3.5
################################################################################################
# Extract a bag of words for a single trace file, where the 
# patterns are all synchronized to a certain event. Meaning that they all start/end with it.
# Takes a <trace-path>:<tid> command line argument and produces a <trace-filename>.BagSyncCluster file.
# Supposedly useful to see the similarities between a set of patterns within a given trace file.
# For now just consider patterns ending with sched_switch 
################################################################################################

from collections import Counter
import pickle
import os.path
import babeltrace
import sys
import statistics
import numpy as np
#from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt1
import time
from leven import levenshtein
# Though the following import is not directly being used, it is required
# for 3D projection to work
#from mpl_toolkits.mplot3d import Axes3D

def lev_dist(X): #returns the levenshtein distance between all pairs of a vector X
    D = np.zeros((len(X),len(X)))
    for row in range(len(X)):
        print('row: ',row,X[row])
        for col in range(row,len(X)):
            D[row][col] = levenshtein(X[row],X[col])
            D[col][row] = D[row][col]
            print('D[',row,',',col,']= ',D[row][col])
    return D

def vmexit_and_irq(X):#returns the exit reasons and irqs of all patterns in the input vector X in the form of a dict ex and irq TODO: use dict comprehension to write code
    ex = {}
    irq = {}
    for i in range(len(X)):
        s = set(X[i].split())
        tmp = [int(st.split('_')[-1]) for st in s if (st.find('kvm_exit')>=0) or (st.find('kvm_x86_exit')>=0)]
        ex.update({i: tmp}) #create a dictionary of pattern index: list of exit codes
        tmp = [int(st.split('_')[-1]) for st in s if (st.find('irq')>=0) and (st.split('_')[-1].isdecimal())]
        irq.update({i: tmp}) #create a dictionary of pattern index: list of irqs
    
    return ex, irq
        
        

def connectivity_matrix(X,ex,irq,thr): #input X: pattern vector, ex: exit reason dict of pattern vector, irq: irq number dict of pattern vector, returns an array representing the connectivity matrix for clustering. thr: is the threshold of similarity between exit reasons and irq numbers that will be accounted as connecting two patterns
    C = np.ones((len(X),len(X))) #default is that all patterns are connected
    for row in range(len(X)):
        print('row: ',row)
        for col in range(len(X)):
            ex_row = set(ex[row]) #get exit reasons for pattern index #row
            ex_col = set(ex[col]) #get exit reasons for pattern index #col
            if (len(ex_row)>0) and (len(ex_col)>0): #if both patterns contain an exit reason
                if float(len(ex_row.intersection(ex_col))) < thr*float(len(ex_row.union(ex_col))): #and if the similarity threshold is violated 
#                    print('exit:',ex_row,ex_col)
                    C[row][col] = 0 #disconnect patterns 
            irq_row = set(irq[row]) #get irq no. for pattern index #row
            irq_col = set(irq[col]) #get irq no. for pattern index #col
            if (len(irq_row)>0) and (len(irq_col)>0): #if both patterns contain an irq no.
                if float(len(irq_row.intersection(irq_col))) < thr*float(len(irq_row.union(irq_col))): #and if the similarity threshold is violated 
#                    print('irq:',irq_row,irq_col)
                    C[row][col] = 0 #disconnect patterns 
    return C


#plt.ion()
#colors = {0: '#000000', 1: '#15b01a', 7: '#0343df', 10: '#9a0eea', 12: '#e50000', 30: '#ff796c', 31: '#ffff14', 32: '#f97306', 48: '#00ffff', 49: '#01ff07', 52: '#75bbfd'}

# a trace collection holds one or more traces
col = babeltrace.TraceCollection()

# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)

duration = []
items = []
corpus=[]
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu','addons_vcpu_enter_guest']

corpus_filename = sys.argv[1].split('kernel')[0]+'vm.pkl'
if (sys.argv[1][-1]=='/'):
    trace_name = sys.argv[1].split('/')[-3]
else:
    trace_name = sys.argv[1].split('/')[-2]
    
print(trace_name)
    
if os.path.exists(corpus_filename) == False: #no corpus file already exists so need to parse the trace
    trace_handle = col.add_trace(sys.argv[1].split(':')[0],'ctf')
    tid = sys.argv[1].split(':')[1]
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

for e in stopwords: #eliminate stop words
    trace = trace.replace(e,'')
    
corpus = trace.split('sched_switch') # Dividing trace so that patterns end with sched_switch

vectorizer = CountVectorizer(ngram_range=(1, 1),token_pattern=r'\b\w+\b',min_df=1, stop_words=stopwords)
X = vectorizer.fit_transform(corpus)

tokens = vectorizer.get_feature_names()
freq = (X.toarray().T).tolist() 
#TODO: just keep non zero entries 
fr = [(x,y) for y in 

f = open(trace_name+'.SyncCount','w')
out = list(map(lambda x,y:[x,y], freq,tokens))
for item in out: #sorted(out, key=lambda x:x[1], reverse=True):
    f.write(str(item)+'\n')
f.close()
        
#d = {tokens[k]:freq[k] for k in range(len(freq))}

tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_train_tf = tf_transformer.transform(X)
    
tokens = vectorizer.get_feature_names()
freq = (X_train_tf.toarray().T).tolist()

f = open(trace_name+'.SyncTf','w')
out = list(map(lambda x,y:[x,y], freq,tokens))
for item in out: #sorted(out, key=lambda x:x[1], reverse=True):
    f.write(str(item)+'\n')
f.close()

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X)
    
tokens = vectorizer.get_feature_names()
freq = (X_train_tfidf.toarray().T).tolist()

f = open(trace_name+'.SyncTfidf','w')
out = list(map(lambda x,y:[x,y], freq,tokens))
for item in out: #sorted(out, key=lambda x:x[1], reverse=True):
    f.write(str(item)+'\n')
f.close()



