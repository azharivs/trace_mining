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

for s in trace_file_names:
    filename = s.splitlines()[0].split(':')[1]
    tids = [int (i) for i in s.splitlines()[0].split(':')[2:]] #generalized to handle multiple VMs per trace file and their tids
    if filename == "": break
    pklfile = filename.split('kernel')[0]+'vm.pkl'
    first = True

    trace_handle = col.add_trace(filename,'ctf')
    if trace_handle is None:
        raise RunTimeError('Cannot add trace')
        
    print('----------------- STARTING CORPUS EXTRACTION: ',filename)
    trace = {i: '' for i in tids} #initialize empty dict of {tid: trace string=''}
    timestamp = {i: [] for i in tids} #initialize empty dict of {tid: event timestamp list=[]}
    index = 0
#    j=0
   # iterate on events
    for event in col.events:
        if event['tid'] in tids: #if event belongs to enlisted tid (qemu runnning the VM)
            tid = event['tid']
            ts = event.timestamp
            if index == 1: 
                start = ts
#            if index >= 10: 
#                break
            if not (event.name in stopwords): index = index + 1
            trace[tid] = trace[tid] + ' ' + event.name.replace('_x86_','_')
            if 'exit_reason' in event: trace[tid] = trace[tid] + '_' + str(event['exit_reason']) 
            if 'vector' in event: trace[tid] = trace[tid] + '_' + str(event['vector'])
            if 'vec' in event: trace[tid] = trace[tid] + '_' + str(event['vec'])
            if 'irq' in event: trace[tid] = trace[tid] + '_' + str(event['irq'])
            timestamp[tid].append(ts)
            if (index % 1000 ==0): #temporarily store in file to free up memory
                print(ts/1000000000, end=' Seconds \r', flush=True)
                if (first): #if it is the first time then don't load the pkl file because there isn't any just dump 
                    f = open(pklfile,'wb')
                    pickle.dump(trace,f)
                    pickle.dump(timestamp,f)
                    f.close()
                    first = False
                else:
                    f = open(pklfile,'rb')
                    trace_old = pickle.load(f)
                    timestamp_old = pickle.load(f)
                    f.close()
                    trace_new = {i: trace_old[i]+trace[i] for i in tids}
                    timestamp_new = {i: timestamp_old[i]+timestamp[i] for i in tids}
#                    print("**************************",j)
#                    print("**************************",j)
#                    for i in tids:
#                        print(i,'::::::',trace_new[i])
#                    for i in tids:
#                        print(i,'::::::',timestamp_new[i])
                    f = open(pklfile,'wb')
                    pickle.dump(trace_new,f)
                    pickle.dump(timestamp_new,f)
                    f.close()
                    trace_new = {}
                    timestamp_new = {}
                    trace_old = {}
                    timestamp_old = {}
#                    j=j+1
#                    if (j==6): break
                trace = {i: '' for i in tids} #reset to empty dict of {tid: trace string=''}
                timestamp = {i: [] for i in tids} #reset to empty dict of {tid: event timestamp list=[]}
                
                
    print("TIDS:", tids)
    f = open(pklfile,'rb')
    trace_old = pickle.load(f)
    timestamp_old = pickle.load(f)
    f.close()
    trace_new = {i: trace_old[i]+trace[i] for i in tids}
    timestamp_new = {i: timestamp_old[i]+timestamp[i] for i in tids}
    f = open(pklfile,'wb')
    pickle.dump(tids,f)
    pickle.dump(trace_new,f)
    pickle.dump(timestamp_new,f)
    f.close()
    
    col.remove_trace(trace_handle)

