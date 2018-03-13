#!/usr/bin/python3.5
from collections import Counter
import babeltrace
import sys


trace_collection = babeltrace.TraceCollection()

# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)
trace_collection.add_trace(sys.argv[1], 'ctf')

for event in trace_collection.events:
    if (event.name == 'lttng_statedump_process_state') :
        print(event['name'],' PID=',event['pid'],'  TID=',event['tid'], ' CPU=',event['cpu'])
        #print(', '.join(event.keys()))
    
