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

first = 0
for s in trace_file_names:
    filename = s.splitlines()[0].split(':')[1]
    tid = int(s.splitlines()[0].split(':')[2])
    if filename == "": break
    if s.find('#') == 0: continue #skip tracefile starting with # for training and only use for test set 
    label = int(s.splitlines()[0].split(':')[0])
    if first == 0:
        first = 1
        labels=np.array(label)
    else:
        labels = np.append(labels,label)
    
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

print('Tokenizing and Vectorizing Corpus ....')
vectorizer = CountVectorizer(ngram_range=(1, 10),token_pattern=r'\b\w+\b',min_df=1, stop_words=stopwords)
X = vectorizer.fit_transform(corpus)

tokens = vectorizer.get_feature_names()
freq = (X.toarray().T).tolist()

f = open(trace_name+'.count','w')
out = list(map(lambda x,y:[x,y], freq,tokens))
for item in out: #sorted(out, key=lambda x:x[1], reverse=True):
    f.write(str(item)+'\n')
f.close()
        
#d = {tokens[k]:freq[k] for k in range(len(freq))}

tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_train_tf = tf_transformer.transform(X)
    
tokens = vectorizer.get_feature_names()
freq = (X_train_tf.toarray().T).tolist()

f = open(trace_name+'.tf','w')
out = list(map(lambda x,y:[x,y], freq,tokens))
for item in out: #sorted(out, key=lambda x:x[1], reverse=True):
    f.write(str(item)+'\n')
f.close()

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X)
    
tokens = vectorizer.get_feature_names()
freq = (X_train_tfidf.toarray().T).tolist()

f = open(trace_name+'.tfidf','w')
out = list(map(lambda x,y:[x,y], freq,tokens))
for item in out: #sorted(out, key=lambda x:x[1], reverse=True):
    f.write(str(item)+'\n')
f.close()

print(labels)

with open(trace_name+'.train','wb') as f:
    pickle.dump(X,f)    
    pickle.dump(vectorizer,f)
    pickle.dump(tf_transformer,f)
    tfidf_transformer.stop_words_ = None
    pickle.dump(tfidf_transformer,f)
    pickle.dump(labels,f)    
