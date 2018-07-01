#!/usr/bin/python3.5
"""
Convert a trace pickle file into graph file represented in NCOL format:

vertex1name vertex2name weight
kvm_entry kvm_pio 3
...


The trace pickles will be read from a list of path .tid files and the graph 
representations will be stored in the same path alongside the pickle files as <tid>.ncol file.
The graph models event adjacency by putting an undirected edge between two consecutive events
with weight equal to the number of times they occur in sequence. 

"""

import pickle
import os.path
from collections import Counter
import logging
from optparse import OptionParser
import sys
from time import time

# parse commandline arguments
op = OptionParser()
op.add_option("--path-file",
              dest="input_filename", type="string",
              help="Take list of trace filenames from this .tid file.")

print(__doc__)
op.print_help()

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

# #############################################################################
# Load trace files from <trace-path-list>.tid file 

input_filename = opts.input_filename

print("Loading trace file list from: ",input_filename)

trace_name = input_filename.split('.')[0]
with open(input_filename,'r') as f:
    trace_file_names = f.readlines()

#stopwords = ['syscall_entry_ioctl','syscall_exit_ioctl','kvm_entry', 'rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu', 'addons_vcpu_enter_guest', 'sched_migrate_task']

stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_stat_runtime', 'addons_vcpu_enter_guest']

#stopwords = ['kvm_entry','kvm_fpu', 'rcu_utilization', 'lttng_statedump_process_state', 'sched_stat_runtime', 'addons_vcpu_enter_guest', 'sched_migrate_task', 'irq_softirq_exit_*', 'irq_softirq_entry_*', 'irq_softirq_raise_*', 'irq_handler_entry_*', 'irq_handler_exit_*', 'block_rq_complete', 'workqueue_activate_work', 'workqueue_queue_work', 'random_mix_pool_bytes_nolock']

for s in trace_file_names:

    if s == "": break
    filename = s.splitlines()[0].split(':')[1]
    if filename == "": break
    tids = [int (i) for i in s.splitlines()[0].split(':')[2:]] #generalized to handle multiple VMs per trace file and their tids
    filename = filename.replace("#","")
    
    pklfile = filename.split('kernel')[0]+'vm.pkl'
    print(pklfile)
    if os.path.exists(pklfile): #there is a corpus file for this trace so read from it
        print('================= READING FROM CORPUS FILE: ',pklfile)        
        f = open(pklfile,'rb')
        tids = pickle.load(f)
        trace = pickle.load(f)
        #timestamps = pickle.load(f) #no need for timestamps at this point
        f.close()
        
        for tid in tids:
            #process vertices
            used = set()
            tmpV=[ss for ss in trace[tid].split() if ss not in used and (used.add(ss) or True)]
            vertices={tmpV[i]:i for i in range(len(tmpV))} #a dict of the form event_name:index for all vertices
            
            #process trace and produce a dict(Counter) of edges ('event1name','event2name'):weight
            #the weight will be the count of this event association within the trace
            #make sure the graph is undirected
            event_list = trace[tid].split()
            weights = Counter() #empty Counter dict object
            event = event_list[1] 
            last_event = event_list[0]
            second_last_event = ''
            for i in range(2,len(event_list)):
                if not (last_event in stopwords): second_last_event = last_event #TODO: I am assuming it never happens that the first event is in stopwords
                if not (event in stopwords): last_event = event #TODO: I am assuming it never happens that the first event is in stopwords
                event = event_list[i]
                if (event in stopwords): continue #ignore stop words
                #if last_event == 'sched_switch': continue #respect execution boundaries and do not put an edge between non adjacent events
                #if event == last_event: #no vertex connects to itself. For now just ignore these patterns.
                #    print(event,': self connecting event ignored!')
                #    continue
                if (last_event != 'sched_switch' and last_event != event):
                    if event < last_event: #sort consecutive events to prevent directed graph
                        v1 = event
                        v2 = last_event
                    else:
                        v2 = event
                        v1 = last_event 
                    weights.update({(v1,v2)}) #increment weight of v1-v2 edge 
                    
                if (second_last_event != 'sched_switch' and second_last_event != event):
                    if event < second_last_event: #sort events to prevent directed graph
                        v1 = event
                        v2 = second_last_event
                    else:
                        v2 = event
                        v1 = second_last_event 
                    #weights.update({(v1,v2)}) #increment weight of v1-v2 edge 

            graphFile = filename.split('kernel')[0]+str(tid)+'.ncol'
            print(graphFile)
            print('================== WRITING OUTPUT OF TID: ',tid,' TO .ncol FILE')
            f = open(graphFile,'wt')
            for e in weights.keys():
                ss = e[0]+' '+e[1]+' '+str(weights[e])+'\n'
                f.write(ss)
            f.close()
    else: print("Trace not found!!!")