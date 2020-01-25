#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
import csv
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count

logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()

    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]

    return subgraph

  def make_undirected(self):

    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)

    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))

    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]:
        self[x].remove(x)
        removed += 1

    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True

    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return order()

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    # remember: self is the graph object we passed from deepwalk_modified.
    # There was no modification of self in the build_deepwalk_corpus func.
    G = self
    if start:
      # ASH:start will be the first node passed from build_deepwalk_corpus, one node
      # ASH:passed in each iteration in build_deepwalk_corpus
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      #ASH: if there is no node in start, just choose one of the nodes from G (the graph
      # that was build using load_edges). We won't use this at the start, as we have a start node
      # IGNORE THIS, we have a start node.
      path = [rand.choice(G.keys())]
    # path_length = 40
    while len(path) < path_length:
      # like pop from stack, get the last value from the path list.
      # path.append() will add a random node from this node's connected nodes, then in the next iteration,
      # this node will become cur, and we'll get a random node from its connected nodes, and so on, till
      # we have a path list of 40 elements OR we land at an unconnected node midway. In this case, we 
      # stop right there, and path has how many ever elements we have until the unconnected node.
      cur = path[-1]
      if len(G[cur]) > 0:
        # if the current paper has a link to another paper, or if another paper links to it,
        # its value will be >0.
        # rand is a random.Random() object (see args). alpha is 0.
        if rand.random() >= alpha:
          # Choose any of the papers in cur's defaultdict list, i.e. one of the papers it is linked to.
          # random() returns [0.0, 1.0), so it will never go to else when alpha is 0 (there are no restarts)
          path.append(rand.choice(G[cur]))
        else: 
          path.append(path[0])
      else:
        # We landed at an unconnected node, break.
        break
    # return a list of 40 or less elements (if we landed at an unconnected node before 40 iterations, we stop)
    # the first element in the list is the 'start' element which is passed to this function.
    return [str(i) for i in path]

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []
  # ASH:numpaths = 10, path_length = 40
  # ASH:list(G.nodes()) makes a list of all the keys of the G defaultdict, i.e. all the 
  # ASH:citing papers (as it is undirectional, this should have all the papers in the file)
  nodes = list(G.nodes())
  # 10 times
  for cnt in range(num_paths):
    # ASH:Shuffle the citing paper ids
    rand.shuffle(nodes)
    for node in nodes:
      # ASH:For each citing paper id, go for a random walk, and add the list of nodes which you reach from 
      # that node. This list can be a maximum of 40 elements long (starting with 'node' in the current 
      # iteration, but can also be less than than if you
      # land at an unconnected node in the middle of the random walk (see the random_walk function for more
      # detailed comments)

      # walks will thus be a list of lists with the current citing paper at the head of each list.
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

  # Return #nodes lists of lists, where each inner list contains the nodes reached by a random walk from 
  # the first node (paper). Remember, there will be 10 sets of lists starting with each node.
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  
  walks = []
  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])

  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])

  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()

  with open(file_) as f:
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
      total = 0
      for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
          adjlist.extend(adj_chunk)
          total += len(adj_chunk)

  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G


def load_edgelist(file_, undirected=True):
  # Build a graph. Graph is an overriden defaultdict.
  G = Graph()
  with open(file_) as f:
    # Skip header
    next(f)
    for l in f:
      # added \t
      #print(l.strip().split('\t'))
      try:
          x, y = l.strip().split('\t')[:2]
      except ValueError:
          # if the row is empty in the file
          print('empty row/spaces')
          continue
      x = int(x)
      y = int(y)
      # G[doc id1] = docid 2. REMEMBER, Graph is a defaultdict, When one paper has 
      # multiple references, they all get added into a list.
      G[x].append(y)
      # Undirected -> so G[doc id2] = docid 1
      if undirected:
        G[y].append(x)

  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G
