#!/usr/bin/env python
"""
Simple command line utility for clustering text.

Reads standard input or reads the lines of the file specified as the first
argument.

Examples
--------
Cluster lines of a file:
>>> ./cluster.py lines.txt

Cluster contents of a directory:
>>> ls | ./cluster.py

Requirements
------------
* numpy
* scikit-learn
"""

from collections import defaultdict
from itertools import izip
from numpy import array
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


"""
DEFAULT VALUES
"""
N = 2  # Size of NGRAMs
N_COMPONENTS = 50  # PCA components
EPSILON = 5.4  # DBSCAN epsilon
MIN_SAMPLES = 3  # DBSCAN min samples


def build_ngrams(names):
    """
    Builds all occurring ngrams in the names

    Parameters
    ----------
    names : list of str
        Strings to list all ngrams in.

    Returns
    -------
    list 
        List of all ngrams.
    """
    ngrams = set()
    for name in names:
        for i in xrange(len(name)-N+1):
            ngrams.add(name[i:i+N])
    return list(ngrams)


def build_features(names):
    """
    Build features for the names

    Parameters
    ----------
    names : list of str
        Strings to build features for.

    Returns
    -------
    numpy.array
        Matrix with features in columns and samples in rows.

    Notes
    -----
    Features built for one set of names are compatible with those generated for
    another set.
    
    Normalizes the data and perform a PCA.
    """
    ngrams = build_ngrams(names)
    A = array  # shortcut
    build = lambda name: A(([float(name.count(ngram)) for ngram in ngrams]))
    X = A(map(build, names))
    # normalize and perform PCA
    X_n = StandardScaler().fit_transform(X)
    X_n = PCA(n_components=N_COMPONENTS).fit_transform(X_n)
    return X_n


def clusternames(names):
    """
    Clusters the names (or any other strings).

    Parameters
    ----------
    names : list of str
        Strings to build features for.

    Returns
    -------
    dict
        Lists of names by cluster number. The -1 key is for names not it any
        cluster.
    """
    X_n = build_features(names)
    # set up and perform dbscan clustering
    dbscan = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES)
    labels = dbscan.fit_predict(X_n)
    # order names by cluster
    ordered = defaultdict(list)
    for label, name in izip(labels, names):
        ordered[label].append(name)
    return ordered


def print_clusters(clustered):
    """
    Print the dict returned by clusternames

    See Also
    --------
    clusternames : takes a dict returned by this function.
    """
    for cluster, contents in clustered.iteritems():
        if cluster != -1:
            print('[*] --- Cluster: %s ---' % contents[0])
        else:
            print('[*] --- Outside any cluster ---')
        for item in contents:
            print('[+] ' + item)
    # Print Number of clusters, ignoring noise if present.
    n_clusters = len(clustered) - (1 if -1 in clustered else 0)
    print('[*] Estimated number of clusters: %d' % n_clusters)


if __name__ == '__main__':
    import fileinput
    names = list(name.strip() for name in fileinput.input())
    clustered = clusternames(names)
    print_clusters(clustered)
