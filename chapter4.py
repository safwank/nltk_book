import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn
from nltk import memoize

def traverse(graph, start, node):
  graph.depth[node.name] = node.shortest_path_distance(start)
  for child in node.hyponyms():
    graph.add_edge(node.name, child.name)
    traverse(graph, start, child)

def hyponym_graph(start):
  G = nx.Graph()
  G.depth = {}
  traverse(G, start, start)
  return G

def graph_draw(graph):
  nx.draw(graph,
   node_size = [16 * graph.degree(n) for n in graph],
   node_color = [graph.depth[n] for n in graph],
   with_labels = False)
  matplotlib.pyplot.show()

def graph_draw_example():
  dog = wn.synset('dog.n.01')
  graph = hyponym_graph(dog)
  graph_draw(graph)

# Dynamic programming examples
def virahanka1(n):
  if n == 0:
    return [""]
  elif n == 1:
    return ["S"]
  else:
    s = ["S" + prosody for prosody in virahanka1(n-1)]
    l = ["L" + prosody for prosody in virahanka1(n-2)]
    return s + l

def virahanka2(n):
  lookup = [[""], ["S"]]
  for i in range(n-1):
    s = ["S" + prosody for prosody in lookup[i+1]]
    l = ["L" + prosody for prosody in lookup[i]]
    lookup.append(s + l)
  return lookup[n]

def virahanka3(n, lookup={0:[""], 1:["S"]}):
  if n not in lookup:
    s = ["S" + prosody for prosody in virahanka3(n-1)]
    l = ["L" + prosody for prosody in virahanka3(n-2)]
    lookup[n] = s + l
  return lookup[n]

@memoize
def virahanka4(n):
  if n == 0:
    return [""]
  elif n == 1:
    return ["S"]
  else:
    s = ["S" + prosody for prosody in virahanka4(n-1)]
    l = ["L" + prosody for prosody in virahanka4(n-2)]
    return s + l
