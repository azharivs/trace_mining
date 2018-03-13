#!/usr/bin/python3.5
import pickle
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
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu']

trace_name = sys.argv[1].split('.')[0]
with open(sys.argv[1],'r') as f:
    trace_file_names = f.readlines()

first = 0
for s in trace_file_names:
    filename = s.splitlines()[0].split(':')[1]
    tid = int(s.splitlines()[0].split(':')[2])
    if filename == "": break

    trace_handle = col.add_trace(filename,'ctf')
    if trace_handle is None:
        raise RunTimeError('Cannot add trace')
        
    print('----------------- STARTING CORPUS EXTRACTION: ',filename)
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
    pklfile = filename.split('kernel')[0]+'vm.pkl'
    f = open(pklfile,'wb')
    duration = float(ts - start)/1000000
    pickle.dump(duration,f)
    pickle.dump(index,f)
    pickle.dump(trace,f)
    f.close()
    
    col.remove_trace(trace_handle)

