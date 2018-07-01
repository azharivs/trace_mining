#!/usr/bin/python3.5
"""
===========================================================
Clustering all threads within trace files using k-means
===========================================================

All threads running in the OS are considered and they are clustered together 
using bag of words approach. Multiple trace files can be used as well.
Both topic extraction view and thread cluster view are generated as output.

This is an example showing how the scikit-learn can be used to cluster
trace files by behavior using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data.

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seems to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as trace data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Seyed Vahid Azhari <azharivs@{iust.ac.ir,gmail.com}>

from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import pickle
import os.path
from collections import Counter
from collections import namedtuple
import babeltrace
import statistics
import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--path-file",
              dest="input_filename", type="string",
              help="Take list of trace filenames from this .tid file.")
op.add_option("--n-clusters",
              dest="true_k", type="int",
              help="Number of clusters to compute.")
op.add_option("--ngram",
              dest="n_gram", type="int",
              help="n-grams used for tokenizing the trace events (number of events to group together).")
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess traces with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from traces.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

col = babeltrace.TraceCollection()

# #############################################################################
# Load trace files from <trace-path-list>.tid file 
# TODO:

input_filename = opts.input_filename

print("Loading trace file list from: ",input_filename)

with open(input_filename,'r') as f:
    trace_file_names = f.readlines()

duration = []
items = []
corpus=[]
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu', 'addons_vcpu_enter_guest']
trace_list=[]
first = 0
threads = {} # dict {(filename,TID): thread-name <TID> filename}
threads_by_index = {} # dict {sequential index: thread-name <TID> filename}
num_threads = [] #list holding number of threads in each trace file
index = 0 #sample (thread) index used as key for threads_by_index dict

for s in trace_file_names:
    if s.find('#') > -1: continue #skip tracefile starting with # 
    if s == "": break
    filename = s.splitlines()[0].split(':')[1]
    if filename == "": break
    tid = int(s.splitlines()[0].split(':')[2])
    lbl = int(s.splitlines()[0].split(':')[0])
    if first == 0:
        first = 1
        target=np.array(lbl)
    else:
        target=np.append(target,lbl)
    filename = filename.replace("#","")
    
    corpus_filename = filename.split('kernel')[0]+'all.pkl'
    trace_name = corpus_filename.split('/')[-2]
    trace_list.append(trace_name)    
    if os.path.exists(corpus_filename): #corpus file already exists
        print('================= READING FROM CORPUS FILE: ',corpus_filename)        
        f = open(corpus_filename,'rb')
        tidlist = pickle.load(f)
        pid = pickle.load(f)
        ppid = pickle.load(f)
        cpu = pickle.load(f)
        name = pickle.load(f)
        trace = pickle.load(f)
        timestamp = pickle.load(f)
        f.close()

        num_threads.append(len(name))
        tmp = {(trace_name,k):name[k]+'<'+str(k)+'> : '+trace_name for k in name.keys()} #append TID and trace filename to process name
        threads.update(tmp)
        corpus = corpus + list(trace.values())
        tmp = {index+k:name[tidlist[k]]+'<'+str(tidlist[k])+'> : '+trace_name for k in range(len(name))} #append TID and trace filename to process name
        index = index + len(name)
        threads_by_index.update(tmp) 
    else: #no corpus file found so ERROR!
        print('NO CORPUS FILE!')
        print('First run ./extract_traces.py over the trace file to produce all.pkl')
    
TraceDataSet = namedtuple("TraceDataSet","corpus threads")
trace_dataset = TraceDataSet(corpus,threads)

print("%d trace files" % len(trace_dataset.corpus))
print("%d categories" % len(trace_dataset.threads))
print()


#labels = trace_dataset.target
########################## Set number of clusters ######################
true_k = opts.true_k #3*np.unique(labels).shape[0]

print("Extracting features from the training traceset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words=stopwords, alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words=stopwords,
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=1.0, max_features=opts.n_features,
                                 min_df=0.0, stop_words=stopwords,ngram_range=(opts.n_gram,opts.n_gram),
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(trace_dataset.corpus)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose, random_state=None)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose, random_state=None)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
#print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
#print("Adjusted Rand-Index: %.3f"
#      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()

#print(trace_dataset.target)
print(km.labels_)

##eq = {i:trace_dataset.target[j] for i in range(true_k) for j in range(len(km.labels_)) if i == km.labels_[j]}
eq = {i: [threads_by_index[j] for j in range(len(threads_by_index)) if i == km.labels_[j]] for i in range(true_k)}
    

if not opts.use_hashing:
    print("Top patterns per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    #print(km.cluster_centers_.shape)
    #print(km.cluster_centers_)
    terms = vectorizer.get_feature_names()
    #print(terms)
    print()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='|.....|')
        print()

    print("Cluster memberships:")
    for i in range(len(eq)):
        print("-----> Cluster %d:" %i, end='')
        for thrd in eq[i]:
            print(thrd.split(':')[0], end='')
        print()
        print()
       


############################## Now test the clustering with those trace entries starting with a '#'
#TODO: do better coding by splitting the train and test data sets all at the beginning