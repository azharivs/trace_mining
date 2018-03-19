#!/usr/bin/python3.5
"""
Multithreaded (parallel) version of get_corpus.py

Input is a .tid file with following format:

Output is a pickle file named vm.pkl in the kernel/ folder 
where the traces are stored with the following format:

    list_of_tids = pickle.load(f) as a list of integers
    trace_dict = pickle.load(f) as a dict of strings indexed by tid
    timestamp_dict = pickle.load(f) as a dict of list of timestamps in nanoseconds and indexed by tid

"""
from multiprocessing.dummy import Pool as ThreadPool 
import pickle
from collections import Counter
import babeltrace
import sys
import statistics
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time

def extract_corpus(tracepath):
    filename = tracepath.splitlines()[0].split(':')[1]
    tids = [int (i) for i in tracepath.splitlines()[0].split(':')[2:]] #generalized to handle multiple VMs per trace file and their tids
    if filename == "": return
    pklfile = filename.split('kernel')[0]+'vm.pkl'
    first = True

    # a trace collection holds one or more traces
    col = babeltrace.TraceCollection()
    trace_handle = col.add_trace(filename,'ctf')
    if trace_handle is None:
        raise RunTimeError('Cannot add trace')
        
    print('----------------- STARTING CORPUS EXTRACTION: ',filename)
    trace = {i: '' for i in tids} #initialize empty dict of {tid: trace string=''}
    timestamp = {i: [] for i in tids} #initialize empty dict of {tid: event timestamp list=[]}
    index = 0
   # iterate on events
    for event in col.events:
        if event['tid'] in tids: #if event belongs to enlisted tid (qemu runnning the VM)
            tid = event['tid']
            ts = event.timestamp
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
                    f = open(pklfile,'wb')
                    pickle.dump(trace_new,f)
                    pickle.dump(timestamp_new,f)
                    f.close()
                    trace_new = {}
                    timestamp_new = {}
                    trace_old = {}
                    timestamp_old = {}
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
    
    return 


# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)

stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu']

trace_name = sys.argv[1].split('.')[0]
with open(sys.argv[1],'r') as f:
    trace_file_names = f.readlines()

pool = ThreadPool(4) 
pool.map(extract_corpus,trace_file_names)
pool.close()
pool.join()
