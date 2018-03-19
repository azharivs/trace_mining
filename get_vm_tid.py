#!/usr/bin/python3.5
from collections import Counter
import babeltrace
import sys
import statistics
import numpy as np
import time
# a trace collection holds one or more traces
col = babeltrace.TraceCollection()

trace_name = sys.argv[1].split('.')[0]
with open(sys.argv[1],'r') as f:
    trace_file_names = f.readlines()

f = open(trace_name+'.tid','w')
for filename in trace_file_names:
    s = filename.splitlines()[0]
    filename = s.split(':')[1]
    if filename == "": break
    trace_handle = col.add_trace(filename,'ctf')
    if trace_handle is None:
        raise RunTimeError('Cannot add trace')
        
    # iterate on events
    tid_list = []
    for event in col.events:
        if (event.name == 'kvm_entry') or (event.name == 'kvm_x86_entry'):#this is the qemu thread running the VM (also works with multiple VMs)
            if not (event['tid'] in tid_list):
                tid_list.append(event['tid']) 

    for tid in tid_list:
        s = s+':'+str(tid)
    
    print(s)    
    f.write(s+'\n')
    col.remove_trace(trace_handle)

f.close()