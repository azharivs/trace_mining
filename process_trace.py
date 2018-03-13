#!/usr/bin/python3
from collections import Counter
import babeltrace
import sys


# a trace collection holds one or more traces
col = babeltrace.TraceCollection()

# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)
if col.add_trace(sys.argv[1], 'ctf') is None:
    raise RuntimeError('Cannot add trace')

# this counter dict will hold execution times:
#
#   task command name -> total execution time (ns)
exec_times = Counter()

# this holds the last `sched_switch` timestamp
last_ts = None
init_ts = None

# iterate on events
for event in col.events:
    # keep only `sched_switch` events
    if event.name != 'sched_switch':
        continue

    # keep only events which happened on CPU 0
    if event['cpu_id'] != 0:
        continue

    # event timestamp
    cur_ts = event.timestamp

    if last_ts is None:
        # we start here
        last_ts = cur_ts
        init_ts = cur_ts
    
    time_diff = cur_ts - init_ts
    if time_diff > 5435106164:
	    continue

    # previous task command (short) name
    prev_comm = event['prev_comm']

    # initialize entry in our dict if not yet done
    if prev_comm not in exec_times:
        exec_times[prev_comm] = 0

    # compute previous command execution time
    diff = cur_ts - last_ts

    # update execution time of this command
    exec_times[prev_comm] += diff

    # update last timestamp
    last_ts = cur_ts

# print top 5
for name, ns in exec_times.most_common(10):
    s = ns / 1000000000
    print('{:20}{} s'.format(name, s))
