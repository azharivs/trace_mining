#!/usr/bin/python3.5
from collections import Counter
import babeltrace
import sys
import statistics
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import time
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

plt.ion()
colors = {0: '#000000', 1: '#15b01a', 7: '#0343df', 10: '#9a0eea', 12: '#e50000', 30: '#ff796c', 31: '#ffff14', 32: '#f97306', 48: '#00ffff', 49: '#01ff07', 52: '#75bbfd'}

# a trace collection holds one or more traces
col = babeltrace.TraceCollection()

# add the trace provided by the user (first command line argument)
# (LTTng traces always have the 'ctf' format)
if col.add_trace(sys.argv[1], 'ctf') is None:
    raise RuntimeError('Cannot add trace')

# this holds the last `kvm_x86_entry` timestamp
entry_flag = 0
entry_ts = None
exit_ts = None
cpu_assigned = None
sojourn = []
vmm_time = []
vect=[None,None]
init_flag = 1
wnd_start = None
wnd_length = 1000000000 #1 sec window
print('STARTING EVENT PROCESSING ...')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# iterate on events
for event in col.events:
    # keep only `kvm exit/entry` events
    if (event.name != 'kvm_x86_entry') and (event.name != 'kvm_x86_exit') and (event.name != 'kvm_userspace_exit'):
        continue
        
    cpu_id = event['cpu_id']

    # event timestamp
    cur_ts = event.timestamp
    
    if wnd_start == None: #initialize start of window time frame
        wnd_start = cur_ts
    
    #vect = [reason_code(=-1 for entries), duration]
    if event.name == 'kvm_x86_entry':
        if entry_flag == 1:
            print('Error! kvm_x86_entry while entry_flag was set')
            exit()
        entry_flag = 1
        entry_ts = cur_ts
        cpu_assigned = cpu_id
        if exit_ts != None:
            diff = entry_ts - exit_ts
            vmm_time.append(diff)
            #vect = [-1.0,diff] 
            
              
    if event.name == 'kvm_x86_exit':
        if entry_flag == 0:
            print('Error! kvm_x86_exit while entry_flag was not set before')
            exit()
        entry_flag = 0
        exit_ts = cur_ts
        if (cpu_assigned != cpu_id):
            print('CPU switched from {:d} at entry to {:d} at exit'.format(cpu_assigned,cpu_id))
        reason_code = event['exit_reason']
        diff = exit_ts - entry_ts
        sojourn.append(diff)
        vect = [reason_code, diff]
        if init_flag:
            X = np.array([vect])
            init_flag = 0
        else:
            X = np.append(X,[vect],axis=0) 
        

#    if event.name == 'kvm_userspace_exit':
#        if entry_flag == 0:
#            print('Error! kvm_userspace_exit while entry_flag was not set before')
#            exit()
#        entry_flag = 0
#        exit_ts = cur_ts
#        if (cpu_assigned != cpu_id):
#            print('CPU switched from at entry to at exit')
#        reason_code = event['exit_reason']
#        diff = exit_ts - entry_ts
#        sojourn.append(diff)
#        vect = [reason_code, 1.0, diff]
#        init_flag = 0
        
    #one full window ready for process and plot
    if cur_ts - wnd_start >= wnd_length: 
        print('====================== NEW WINDOW REPORT ===========================')
        avg_cycle_time = float(sum(sojourn)+sum(vmm_time))/len(sojourn)

        vmm_nrm = [float(i)/avg_cycle_time for i in vmm_time]
        avg_vmm = statistics.mean(vmm_nrm)
        stdev_vmm = statistics.stdev(vmm_nrm)

        #signature = np.array([[-1.0,avg_vmm,stdev_vmm]],dtype=float) #this is for the previous 2D plots without the probability of exit code
        signature = np.array([[None, None, None, None]]) # [reason code, avg vm dwelling time, stdev vm dwelling time, probability of this exit code]
        cc = [] #bad way of populating colors

        num_samples = np.size(X,axis=0)
        for i in range(64): #iterate over exit reason codes
            if (X[X[:,0]==i]).any(): #if there is any exit code == i then proceed 
                ratio = float((X[:,0]==i).sum())/num_samples 
                avg_vm_time = np.mean(X[X[:,0]==i,1:])/avg_cycle_time
                stdev_vm_time = np.std(X[X[:,0]==i,1:])/avg_cycle_time
                if (signature == None).any():
                    signature = np.array([[float(i),avg_vm_time,stdev_vm_time,ratio]])
                else:
                    signature = np.append(signature,[[float(i),avg_vm_time,stdev_vm_time,ratio]],axis=0)
                cc.append(colors[i]) #use the corresponding color: FIXME: if there are more exit codes that I have through of then we are in trouble. Need to fix the colors list at the begining of the code to correct this

        sojourn_nrm = [float(i)/avg_cycle_time for i in sojourn]
        vmm_nrm = [float(i)/avg_cycle_time for i in vmm_time]
        avg_sojourn = statistics.mean(sojourn_nrm)
        stdev_sojourn = statistics.stdev(sojourn_nrm)
        avg_vmm = statistics.mean(vmm_nrm)
        stdev_vmm = statistics.stdev(vmm_nrm)
        sj_arr = np.array(sojourn_nrm).reshape(-1,1)
        #kmeans = KMeans(n_clusters=8, random_state=0).fit(sj_arr)
        print('Number of times VM was executed = {0:3d}'.format(len(sojourn)))
        print('Average fraction VM sojourn time = {0:.3f} usec and {0:.3f} vmm time'.format(avg_sojourn,avg_vmm))        
        print('Stdev fraction VM sojourn time = {0:.3f} usec'.format(stdev_sojourn))        
        print('Stdev fraction VMM time = {0:.3f} usec'.format(stdev_vmm))        

        #print(kmeans.labels_)
        #print(kmeans.cluster_centers_)

        #labels = kmeans.labels_
        #plt.scatter(labels,sj_arr)
        #plt.hist(sj_arr,50,normed=1)
        #plt.show()

        plt.hold(False)
        #plt.scatter(signature[:,1],signature[:,2],s=100*(1-np.log10(signature[:,3])),c=cc)      
        ax.scatter(signature[:,1],signature[:,2],signature[:,3],s=100,c=cc)
        plt.grid(b=True)
        plt.xlabel('Mean Normalized Dwelling Time')
        plt.ylabel('Stdev Normalized Dwelling Time')
        ax.set_zlabel('Probability of Exit Reason')
        plt.hold(True)
        plt.title(time.ctime(wnd_start/1000000000))
        ax.text(2,2,1,'Labels and colors correspond to exit codes')
        for i in signature[:,0]: #this was for the previous 2D plot
            ax.text(np.float(signature[signature[:,0]==i,1]),np.float(signature[signature[:,0]==i,2]),np.float(signature[signature[:,0]==i,3]),str(i))
            #ax.text(signature[0,1],signature[0,2],signature[0,3],str(i))
        plt.xlim(-0.0,2.0)
        plt.ylim(-0.0,2.0)
        ax.set_zlim(-0.0,1.0)
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

        sojourn = []
        vmm_time = []
        vect = [None,None]
        init_flag = 1
        wnd_start = cur_ts
        # end of processing for latest window of samples


