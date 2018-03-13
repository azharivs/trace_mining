#!/usr/bin/python3.5
"""
=======================================
Clustering trace files using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
trace files by topics using a bag-of-words approach. This example uses
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
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Seyed Vahid Azhari <azharivs@iust.ac.ir>

from __future__ import print_function
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()



n_top_words = 20

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--path-file",
              dest="input_filename", type="string",
              help="Take list of trace filenames from this .tid file.")
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

trace_name = input_filename.split('.')[0]
with open(input_filename,'r') as f:
    trace_file_names = f.readlines()

duration = []
items = []
corpus=[]
stopwords = ['rcu_utilization', 'lttng_statedump_process_state', 'sched_waking', 'sched_stat_wait', 'sched_stat_runtime', 'timer_hrtimer_cancel', 'timer_hrtimer_expire_exit', 'timer_cancel', 'kvm_fpu', 'addons_vcpu_enter_guest']
trace_list=[]
first = 0

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
    trace_list.append(filename)    
    
    corpus_filename = filename.split('kernel')[0]+'vm.pkl'
    if os.path.exists(corpus_filename) == False: #no corpus file already exists so need to parse the trace
        trace_handle = col.add_trace(filename,'ctf')
        if trace_handle is None:
            raise RunTimeError('Cannot add trace')
        
        print('----------------- STARTING EVENT PROCESSING: ',filename)
        trace=''
        index = 0
       # iterate on events
        for event in col.events:
            if event['tid'] == tid:
                ts = event.timestamp
                if index == 1: 
                    start = ts
                #if index >= 100: 
                #    break
                if not (event.name in stopwords): index = index + 1
                trace = trace + ' ' + event.name.replace('_x86_','_')
                if 'exit_reason' in event: trace = trace + '_' + str(event['exit_reason']) 
                if 'vector' in event: trace = trace + '_' + str(event['vector'])
                if 'vec' in event: trace = trace + '_' + str(event['vec'])
                if 'irq' in event: trace = trace + '_' + str(event['irq'])
        dur = float(ts - start)/1000000
        col.remove_trace(trace_handle)
    else: #there is a corpus file for this trace so read from it
        print('================= READING FROM CORPUS FILE: ',corpus_filename)        
        f = open(corpus_filename,'rb')
        dur = pickle.load(f)
        index = pickle.load(f)
        trace = pickle.load(f)
        f.close()
        
    duration.append(dur)
    items.append(index)    
    corpus.append(trace)

TraceDataSet = namedtuple("TraceDataSet","corpus target_names target")
trace_dataset = TraceDataSet(corpus,trace_list,target)

print("%d trace files" % len(trace_dataset.corpus))
print("%d categories" % len(trace_dataset.target_names))
print()

labels = trace_dataset.target
true_k = 3*np.unique(labels).shape[0]

print("Extracting features from the training traceset using a sparse vectorizer")
t0 = time()

tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=opts.n_features,
                                 min_df=0.0, stop_words=stopwords,ngram_range=(1,1))
tfidf = tfidf_vectorizer.fit_transform(trace_dataset.corpus)


tf_vectorizer = CountVectorizer(max_df=1.0, max_features=opts.n_features,
                                 min_df=0.0, stop_words=stopwords,ngram_range=(1,1))
tf = tf_vectorizer.fit_transform(trace_dataset.corpus)

n_samples,n_features = tfidf.shape
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % tfidf.shape)
print()

t0 = time()
nmf = NMF(n_components=opts.n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Fit the NMF model
print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
      "tf-idf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=opts.n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=opts.n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)