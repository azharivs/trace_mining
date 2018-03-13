#!/usr/bin/python3.5
from collections import Counter
import babeltrace
import sys
import statistics
import numpy as np
#from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
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
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu']
with open(sys.argv[1],'r') as f:
    trace_file_names = f.readlines()
    print(trace_file_names)

for filename in trace_file_names:
    filename = filename.splitlines()[0]
    print(filename)
    trace_handle = col.add_trace(filename,'ctf')
    if trace_handle is None:
        raise RunTimeError('Cannot add trace')
        
    print('STARTING EVENT PROCESSING: ',filename)
    trace=''
    index = 0
   # iterate on events
    for event in col.events:
        if event['tid'] == 4330: #3513: #7202:
            ts = event.timestamp
            if index == 1: 
                start = ts
            #if index >= 100000: 
            #    break
            if not (event.name in stopwords): index = index + 1
            trace = trace + ' ' + event.name
            if 'exit_reason' in event: trace = trace + '_' + str(event['exit_reason']) 
            if 'vector' in event: trace = trace + '_' + str(event['vector'])
            if 'vec' in event: trace = trace + '_' + str(event['vec'])
            if 'irq' in event: trace = trace + '_' + str(event['irq'])

    duration.append(float(ts - start)/1000000)
    items.append(index)    
    corpus.append(trace)
    col.remove_trace(trace_handle)


vectorizer = CountVectorizer(ngram_range=(1, 10),token_pattern=r'\b\w+\b',min_df=1, stop_words=stopwords)
X = vectorizer.fit_transform(corpus)

tokens = vectorizer.get_feature_names()
freq = X.toarray() 
#freq = np.divide(X.toarray(),10000) 
max_freq = np.amax(freq,axis=0)

#print(tokens)
#print(freq)
#print(max_freq)
traces,patterns = freq.shape

for i in range(0,traces):
    for j in range(i+1,traces):
        diff = np.divide(np.abs(freq[i]-freq[j]),max_freq)
        #significants = list(map(lambda x,y: , tokens,diff))
        significants = [(tokens[k],diff[k],freq[i,k],freq[j,k],max_freq[k]) for k in range(0,patterns) if (diff[k] > 0.5) and (freq[i,k]+freq[j,k] > 1000 ) ]
        #significants = {tokens[k]:diff[k] for k in range(0,patterns) if (diff[k] > 0.5) and (freq[i,k]+freq[j,k] > 100 ) }
        print('******************************************************************')
        print('COMPARING TRACES',trace_file_names[i],':',duration[i],'msec, ',items[i],' events \n                ',trace_file_names[j],':',duration[j],'msec, ',items[j],' events')
        print('******************************************************************')
        
        for p in sorted(significants, key=lambda x: x[1], reverse=True) :
            print(p) 



#freq = (X.toarray())[0].tolist()

#out = list(map(lambda x,y:(x,y), tokens,freq))
#for item in out:
#    print(item)