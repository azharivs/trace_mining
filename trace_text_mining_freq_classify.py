#!/usr/bin/python3.5
# This is a frequency based approach to text classification as opposed 
# to a pattern based approach using levenshtein's distance
from collections import Counter
import babeltrace
import sys
import statistics
import numpy as np
#from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt1
import time
from leven import levenshtein
# Though the following import is not directly being used, it is required
# for 3D projection to work
#from mpl_toolkits.mplot3d import Axes3D

def lev_dist(X): #returns the levenshtein distance between all pairs of a vector X
    D = np.zeros((len(X),len(X)))
    for row in range(len(X)):
        for col in range(len(X)):
            D[row][col] = levenshtein(X[row],X[col])
    return D

def vmexit_and_irq(X):#returns the exit reasons and irqs of all patterns in the input vector X in the form of a dict ex and irq TODO: use dict comprehension to write code
    ex = {}
    irq = {}
    for i in range(len(X)):
        s = set(X[i].split())
        tmp = [int(st.split('_')[-1]) for st in s if (st.find('kvm_exit')>=0) or (st.find('kvm_x86_exit')>=0)]
        ex.update({i: tmp}) #create a dictionary of pattern index: list of exit codes
        tmp = [int(st.split('_')[-1]) for st in s if (st.find('irq')>=0) and (st.split('_')[-1].isdecimal())]
        irq.update({i: tmp}) #create a dictionary of pattern index: list of irqs
    
    return ex, irq
        
        

def connectivity_matrix(X,ex,irq,thr): #input X: pattern vector, ex: exit reason dict of pattern vector, irq: irq number dict of pattern vector, returns an array representing the connectivity matrix for clustering. thr: is the threshold of similarity between exit reasons and irq numbers that will be accounted as connecting two patterns
    C = np.ones((len(X),len(X))) #default is that all patterns are connected
    for row in range(len(X)):
        for col in range(len(X)):
            ex_row = set(ex[row]) #get exit reasons for pattern index #row
            ex_col = set(ex[col]) #get exit reasons for pattern index #col
            if (len(ex_row)>0) and (len(ex_col)>0): #if both patterns contain an exit reason
                if float(len(ex_row.intersection(ex_col))) < thr*float(len(ex_row.union(ex_col))): #and if the similarity threshold is violated 
#                    print('exit:',ex_row,ex_col)
                    C[row][col] = 0 #disconnect patterns 
            irq_row = set(irq[row]) #get irq no. for pattern index #row
            irq_col = set(irq[col]) #get irq no. for pattern index #col
            if (len(irq_row)>0) and (len(irq_col)>0): #if both patterns contain an irq no.
                if float(len(irq_row.intersection(irq_col))) < thr*float(len(irq_row.union(irq_col))): #and if the similarity threshold is violated 
#                    print('irq:',irq_row,irq_col)
                    C[row][col] = 0 #disconnect patterns 
    return C


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

skip_trace = False

#trace_name will be used as base string for naming of various output files
if (sys.argv[1].find('.patterns') >= 0):
    trace_name = sys.argv[1].split('.')[0]
else:
    tmp = sys.argv[1].split('/')
    if tmp[-1] == '': 
        trace_name = tmp[-3]
    else:
        trace_name = tmp[-2]

if (len(sys.argv) == 2):  #if only file name as argument with no other directive then perform trace processing
    skip_trace = False
else: 
    if (sys.argv[2] == 'no-trace'): #if no-trace option is used then skip trace processing part otherwise and load the processed output n-grams
        skip_trace = True
print('Trace base name:',trace_name)
        
if (not skip_trace): #process trace file
    
    if col.add_trace(sys.argv[1], 'ctf') is None:
        raise RuntimeError('Cannot add trace')

        
    print('STARTING EVENT PROCESSING: ',sys.argv[1])
    trace=''
    index = 0
    # iterate on events
    for event in col.events:      
        if event['tid'] == 18253: # 4330: #3513: #7202:
            ts = event.timestamp
            if index == 1: 
                start = ts
            if index >= 1000: 
                break
            if not (event.name in stopwords): index = index + 1
            trace = trace + ' ' + event.name.replace('_x86_','_')
            if 'exit_reason' in event: trace = trace + '_' + str(event['exit_reason']) 
            if 'vector' in event: trace = trace + '_' + str(event['vector'])
            if 'vec' in event: trace = trace + '_' + str(event['vec'])
            if 'irq' in event: trace = trace + '_' + str(event['irq'])

    duration.append(float(ts - start)/1000000)
    items.append(index)    
    corpus.append(trace)

    vectorizer = CountVectorizer(ngram_range=(1, 10),token_pattern=r'\b\w+\b',min_df=1, stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)

    tokens = vectorizer.get_feature_names()
    freq = (X.toarray())[0].tolist()

    f = open(trace_name+'.count','w')
    out = list(map(lambda x,y:(x,y), tokens,freq))
    for item in out:
        f.write(str(item)+'\n')
    f.close()
        
    d = {tokens[k]:freq[k] for k in range(len(freq))}

    tf_transformer = TfidfTransformer(use_idf=False).fit(X)
    X_train_tf = tf_transformer.transform(X)
    
    tokens = vectorizer.get_feature_names()
    freq = (X_train_tf.toarray())[0].tolist()

    f = open(trace_name+'.tf','w')
    out = list(map(lambda x,y:(x,y), tokens,freq))
    for item in out:
        f.write(str(item)+'\n')
    f.close()

    tfidf_transformer = TfidfTransformer(use_idf=True)
    X_train_tfidf = tfidf_transformer.fit_transform(X)
    
    tokens = vectorizer.get_feature_names()
    freq = (X_train_tfidf.toarray())[0].tolist()

    f = open(trace_name+'.tfidf','w')
    out = list(map(lambda x,y:(x,y), tokens,freq))
    for item in out:
        f.write(str(item)+'\n')
    f.close()

#at this point data is to be prepared for processing either from processed n-gram file  
else: 
    print("Reading from .patterns file")
    
    with open(trace_name+'.patterns','r') as f:
        pattern_list = f.readlines()

    d={}
    tokens = []
    freq = []

    for pattern in pattern_list: #create dictionary from patterns  {'pattern': frequency}
        f = int(pattern.splitlines()[0].replace("'","").replace("(","").replace(")","").split(',')[1])
        t = pattern.splitlines()[0].replace("'","").replace("(","").replace(")","").split(',')[0]
        d.update({t:f})
        freq.append(f)
        tokens.append(t)
    
#Now everything is in the same variables so this part is executed regardless of reading from trace file or pattern file



top = Counter(d).most_common(100) #get Counter (dict) of top 100 patterns
top_patterns = list(map(lambda x:x[0] , top)) #just get the patterns part (keys and not values)
X=np.array(top_patterns)
