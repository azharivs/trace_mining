#!/usr/bin/python3.5
"""
Converts old single thread (VM) format of vm.pkl file to new one

Output is a pickle file named vm.pkl in the kernel/ folder 
where the traces are stored with the following format:

    list_of_tids = pickle.load(f) as a list of integers
    trace_dict = pickle.load(f) as a dict of strings indexed by tid
    timestamp_dict = pickle.load(f) as a dict of list of timestamps in nanoseconds and indexed by tid

"""
from multiprocessing import Pool #for multi processing
from multiprocessing.dummy import Pool as ThreadPool  #its equivalent for multi threading
import os.path
import pickle
from collections import Counter
import sys
import statistics
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time

def convert_pkl(tracepath):
    if tracepath == "": return
    filename = tracepath.splitlines()[0].split(':')[1]
    if filename == "": return
    tids = [int (i) for i in tracepath.splitlines()[0].split(':')[2:]] #generalized to handle multiple VMs per trace file and their tids
    pklfile = filename.split('kernel')[0]+'vm.pkl'
    first = True
    if (os.path.exists(pklfile+'.tmp')):
        print("Already in converted form:",pklfile)
        return #already in new converted form
    if not (os.path.exists(pklfile)):
        print("No ",pklfile, " run :")
        print("./get_corpus_parallel.py with apropriate input file")
        return  #no vm.pkl file need to create one using get_corpus_parallel.py

    print('----------------- STARTING CORPUS CONVERSION: ',pklfile)
    f = open(pklfile,'rb') 
    duration = pickle.load(f)
    items = pickle.load(f)
    trace_old = pickle.load(f)
    trace_new = {tids[0]: trace_old}
    timestamp_new = {tids[0]: []}
    f.close()
    f = open(pklfile,'wb')
    pickle.dump(tids,f)
    pickle.dump(trace_new,f)
    pickle.dump(timestamp_new,f)
    f.close()

    print("------- done with ",pklfile, "TID:", tids)
    return 


# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)

stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu']

trace_name = sys.argv[1].split('.')[0]
with open(sys.argv[1],'r') as f:
    trace_file_names = f.readlines()

pool = Pool(4) #Multi processing seems a better option for this script
#pool = ThreadPool(8) 
pool.map(convert_pkl,trace_file_names)
pool.close()
pool.join()
