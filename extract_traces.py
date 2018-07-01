#!/usr/bin/python3.5
"""
This script extracts all traces of a trace file and groups them according to their tid.
It then stores the resulting structure as a pickle file by the name <all.pkl> in the <kernel> folder 
of the trace directory tree.

Note: The input file name should contain a list of trace paths ending in a colon : for each line

The structure of the stored data is a set of dicts and one list with following semantics:
------------------
tidlist 
pid as dict {tid:pid}
ppid (parent pid) as dict {tid:parent pid}
cpu assigned as dict {tid:cpu}
name of the process as dict {tid:name}
the entire aggregated trace string as dict {tid:trace string}
list of timestamps where each timestamp corresponds to an event stored in the trace string as dict {tid:[t1,t2,t3,...]} ti are in nano seconds
-------------------

"""

from __future__ import print_function
import pickle
import os.path
from collections import Counter
from collections import namedtuple
import babeltrace
import sys
import statistics
import numpy as np
import time
from optparse import OptionParser

# parse commandline arguments
op = OptionParser()
op.add_option("--path-file",
              dest="input_filename", type="string",
              help="Take list of trace filenames from this .tid file.") #the tid values will be ignored because all tids will be extracted

print(__doc__)
op.print_help()

argv = sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

# a trace collection holds one or more traces
col = babeltrace.TraceCollection()

# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)

duration = []
items = []
corpus=[]
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu']

trace_name = opts.input_filename.split('.')[0]
with open(opts.input_filename,'r') as f:
    trace_file_names = f.readlines()

first = 1
for s in trace_file_names:
    filename = s.splitlines()[0].split(':')[1]
    if filename == "": break

    first = 1
    trace_handle = col.add_trace(filename,'ctf')
    if trace_handle is None:
        raise RunTimeError('Cannot add trace')
    
    print('----------------- PROCESSING TRACE: ',filename)
    
    tidlist = []    
    index = 0
    for event in col.events:
#        if index >= 100: 
#            break
        if event.name == 'lttng_statedump_process_state': #lttng events that specify process information
            if first == 1:
                first = 0
                tid = event['tid']
                tidlist = [tid]
                pid = {tid:event['pid']}
                ppid = {tid:event['ppid']}
                cpu = {tid:event['cpu']}
                name = {tid:event['name']}
                trace = {tid:''}
                timestamp = {tid:[]}
            else:
                tid = event['tid']
                tidlist.append(tid)
                pid.update({tid:event['pid']})
                ppid.update({tid:event['ppid']})
                cpu.update({tid:event['cpu']})
                name.update({tid:event['name']})
                trace.update({tid:''})
                timestamp.update({tid:[]})
            print('----------------- COLLECTING PROCESS/THREAD INFORMATION: TID<',tid,'> PID<',pid[tid],'> NAME<',name[tid],'>')
            
        elif (event.name.find('lttng') == -1) and (event['tid'] in tidlist): #other non-lttng events that need collecting and have been encountered before (i.e., we have their tid)
          
            if not (event.name in stopwords): index = index + 1
            tid = event['tid']
            trace[tid] = trace[tid] + ' ' + event.name.replace('_x86_','_')
            if 'exit_reason' in event: trace[tid] = trace[tid] + '_' + str(event['exit_reason']) 
            if 'vector' in event: trace[tid] = trace[tid] + '_' + str(event['vector'])
            if 'vec' in event: trace[tid] = trace[tid] + '_' + str(event['vec'])
            if 'irq' in event: trace[tid] = trace[tid] + '_' + str(event['irq'])
            timestamp[tid].append(event.timestamp)
            
    
    
    pklfile = filename.split('kernel')[0]+'all.pkl'
    print('------------------ WRITING OUTPUT TO ',pklfile)
    f = open(pklfile,'wb')
    pickle.dump(tidlist,f)
    pickle.dump(pid,f)
    pickle.dump(ppid,f)
    pickle.dump(cpu,f)
    pickle.dump(name,f)
    pickle.dump(trace,f)
    pickle.dump(timestamp,f)
    f.close()
    
    col.remove_trace(trace_handle)

