#!/usr/bin/python3.5
"""
Writes output to leven.diff

"""
import pickle
import os.path
from collections import Counter
import babeltrace
import sys
import statistics
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
import time
from leven import levenshtein

# a trace collection holds one or more traces
col = babeltrace.TraceCollection()

duration = []
items = []
corpus=[]
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu', 'addons_vcpu_enter_guest']
trace_list=[]
first = 0

with open(sys.argv[1],'r') as f:
    trace_file_names = f.readlines()

for s in trace_file_names:
    if s.find('#') > -1: continue #skip tracefile starting with # 
    if s == "": break
    filename = s.splitlines()[0].split(':')[1]
    if filename == "": break
    tid = int(s.splitlines()[0].split(':')[2])
    lbl = int(s.splitlines()[0].split(':')[0])
    if first == 0:
        first = 1
        target=np.array(lbl)
    else:
        target=np.append(target,lbl)
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


print("Starting Vectorization ...")

vectorizer = CountVectorizer(ngram_range=(1, 1),token_pattern=r'\b\w+\b',min_df=1, stop_words=stopwords)
X = vectorizer.fit_transform(corpus)

tokens = vectorizer.get_feature_names()
freq = X.toarray() 
freq = np.divide(X.toarray(),10000) 
max_freq = np.amax(freq,axis=0)

#print(tokens)
#print(freq)
#print(max_freq)
traces,patterns = freq.shape

f = open('leven.diff','w')

for i in range(0,traces):
    for j in range(i+1,traces):
        print('******************************************************************')
        f.write('******************************************************************')
        print('COMPARING TRACES',trace_file_names[i],':',duration[i],'msec, ',items[i],' events \n                ',trace_file_names[j],':',duration[j],'msec, ',items[j],' events')
        f.write('COMPARING TRACES'+trace_file_names[i]+':'+str(duration[i])+'msec, '+str(items[i])+' events \n                '+trace_file_names[j]+':'+str(duration[j])+'msec, '+str(items[j])+' events')
        lv = levenshtein(corpus[i],corpus[j])
        print('Levenstein Distance=',lv)
        f.write('Levenstein Distance='+str(lv))
        print('******************************************************************')
        f.write('******************************************************************')
        

f.close()

#freq = (X.toarray())[0].tolist()

#out = list(map(lambda x,y:(x,y), tokens,freq))
#for item in out:
#    print(item)