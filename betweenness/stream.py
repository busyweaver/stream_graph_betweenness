# Copyright (C) 2017-2020 Léo Rannou - Sorbonne Université/LIP6 - Thales
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# OVERALL Todos:
# TODO : Harmoniser les noms des fonctions, augmented ou segmented dès que temporal node au lieu de node
import copy
import csv
import datetime as dt
import functools as f
import itertools
import json
import math
import matplotlib.dates as md
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib.font_manager
import networkx as nx
#import networkit as nk
import numpy
import numpy.polynomial.polynomial as nppol
from operator import itemgetter
import pandas as pd
import random
import time
from collections import Counter, defaultdict
from joblib import Parallel, delayed
from matplotlib import animation
from matplotlib import cm
from sortedcontainers import SortedSet
import heapq
from straph import components as cmp
from straph import etf
from straph.paths import paths as ap
from straph.paths import meta_walks as mw
from straph.utils import get_cmap

from straph import ordgraphdis as org
from straph import fibheap as fib
#from fibheap import *
#from straph import ordgraphcon as con

def DFS_iterative(v, Neighborhood):
    """
    Performs a Depth First search from the temporal node *v*

    :param v: Source node for the DFS, a temporal node (t0,t1,u)
    :param Neighborhood: Neighborhood of the Stream Graph : (t0,t1,u) -> [(t0,t1,w),....]
    :return: The weakly connected component containing *v* along with visited nodes (other temporal nodes in the wcc)
    """
    visited = set()
    Q = [v]
    component = []
    while len(Q) > 0:
        v = Q.pop()
        if v not in visited:
            visited.add(v)
            component.append(v)
            for w in Neighborhood[v]:
                Q.append(w)
    return component, visited


def stream_graph_from_events_list(list_events, stream_id=None):
    """
    Construct a ``StreamGraph`` from a list of events

    :param list_events:
    :param stream_id:
    :return:
    """
    E = defaultdict(list)
    W = defaultdict(list)

    nodes_to_label = {}
    label_to_id = defaultdict(lambda: len(label_to_id))
    min_t, max_t = math.inf, -math.inf
    for e in list_events:
        c = e[0]
        if c == 2:
            _, t0, t1, u_label = e
            u = label_to_id[u_label]
            nodes_to_label[u] = u_label
            min_t, max_t = min(min_t, t0), max(max_t, t1)
            W[u] += [t0, t1]
        if c == 1:
            _, t0, t1, u_label, v_label = e
            u = label_to_id[u_label]
            v = label_to_id[v_label]
            if (v, u) in E:
                l = (v, u)
            else:
                l = (u, v)
            E[l] += [t0, t1]
    S = StreamGraph(times=[min_t, max_t],
                    nodes=list(W.keys()),
                    links=list(E.keys()),
                    node_presence=[W[k] for k in W.keys()],
                    link_presence=[E[k] for k in E.keys()],
                    node_to_label=nodes_to_label,
                    node_to_id={i: i for i in W.keys()},
                    id=stream_id)
    return S


##############################################################
#           Postprocess Communities                          #
##############################################################

def merge_clusters_into_communities(sdag, cluster_id_to_community, prev_id, next_id,
                                    cnt_com, alpha, done_clusters):
    """
    Merge Two Adjacent Clusters into a same community if they share an important number of nodes

    :param sdag:
    :param cluster_id_to_community:
    :param prev_id:
    :param next_id:
    :param cnt_com:
    :param alpha:
    :param done_clusters:
    :return:
    """
    prev_comp = sdag.c_nodes[prev_id]
    next_comp = sdag.c_nodes[next_id]
    # print("Id comp :",prev_id, " numb clusters :",len(prev_comp.clusters))
    # print("Next id comp :",next_id, " numb clusets :",len(next_comp.clusters))
    for p_i, p_c in enumerate(prev_comp.clusters):
        assert (prev_comp.id, p_i) in cluster_id_to_community
        for n_i, n_c in enumerate(next_comp.clusters):
            if (next_comp.id, n_i) not in done_clusters:
                intersec = len(n_c & p_c)
                if intersec and intersec / len(n_c | p_c) >= alpha:
                    # print("MERGER :",len(n_c & p_c)/ len(n_c | p_c))
                    cluster_id_to_community[(next_comp.id, n_i)] = cluster_id_to_community[(prev_comp.id, p_i)]
                    done_clusters.add((next_comp.id, n_i))
                # else:
                # print("NOT MERGER :",len(n_c & p_c)/ len(n_c | p_c))
    for n_i, n_c in enumerate(next_comp.clusters):
        if (next_comp.id, n_i) not in done_clusters:
            cluster_id_to_community[(next_comp.id, n_i)] = cnt_com

            cnt_com += 1
    return cnt_com


def postprocess_communities(sdag, alpha=0.70):
    tmp_done_couples = set()
    # BFS on Sdag to progressively merge Communities
    cluster_id_to_community = {}
    unvisited = sorted([(sc.times[0], sc.id) for sc in sdag.c_nodes], reverse=True)
    a_l = sdag.adjacency_list()

    cnt_com = 0
    done_clusters = set()  # id clusters attributed to a community

    while len(unvisited) != 0:
        _, c_id = unvisited.pop()
        # print("len unvisited :", len(unvisited))
        comp = sdag.c_nodes[c_id]
        for i, c_c in enumerate(comp.clusters):
            if (c_id, i) not in cluster_id_to_community:  # Set unattributed clusters to a community
                cluster_id_to_community[(c_id, i)] = cnt_com
                cnt_com += 1
        if c_id in a_l and a_l[c_id]:
            # Merge Components
            for next_id in a_l[c_id]:
                tmp_done_couples.add((c_id, next_id))
                cnt_com = merge_clusters_into_communities(sdag,
                                                          cluster_id_to_community,
                                                          prev_id=c_id,
                                                          next_id=next_id,
                                                          cnt_com=cnt_com,
                                                          alpha=alpha,
                                                          done_clusters=done_clusters)

    communities = {i: [] for i in set(cluster_id_to_community.values())}  # List of communities
    # print("Set Community ids :",set(cluster_id_to_community.values()))
    for k, v in cluster_id_to_community.items():
        # print(" Cluster id :",k)
        # print(" Community id :",v)
        # k = (comp.id, pos of a cluster in list)
        # get comp corresponding to cluster:
        comp = sdag.c_nodes[k[0]]
        # get cluster corresponding to cluster id:
        clust = comp.clusters[k[1]]
        communities[v] += [(comp.times[0], comp.times[1], n) for n in clust]
    communities = postprocess_clusters(communities)
    return [v for v in communities.values()]


##############################################################
#           Dict_clusters to Signals                         #
##############################################################

def postprocess_clusters(dict_clusters):
    """
    Merge adjacent clusters withe the same value

    :param dict_clusters:  Dictionnary : prop -> clusters
    :return:
    """
    # 1. For each prop value sort the cluster by (node,t0)
    # 2. Browse the  [(t0,t1,u),..] and keep a min(t0),max(t1) until encountering a t1!=t0
    dict_clusters_pp = {k: [] for k in dict_clusters}
    for k, cluster in dict_clusters.items():
        cluster = sorted(cluster, key=lambda x: (x[2], x[0]))
        t0_current, t1_current, prev_u = cluster[0]
        for t0, t1, u in cluster[1:]:
            if u == prev_u and t0 == t1_current:  # previous temporal node equals to current one
                t1_current = t1
            else:
                dict_clusters_pp[k].append((t0_current, t1_current, prev_u))
                prev_u = u
                t0_current, t1_current = t0, t1
        dict_clusters_pp[k].append((t0_current, t1_current, prev_u))
    return dict_clusters_pp


def clusters_to_signals(S, dict_clusters, datetime=False, withnan=True):
    """
    Process clusters (a list of temporal nodes [(t0,t1,u),...]) into a signal (a pandas Series)

    :param withnan:
    :param datetime:
    :param S:
    :param dict_clusters:
    :return:
    """
    # times_index = pd.Index(sorted([v[0] for l in dict_clusters.values() for v in l]
    #                               + [v[1] for l in dict_clusters.values() for v in l]))

    nodes_to_time_to_value = defaultdict(dict)
    # As intervals can overlap we need to define an eps for the signal !!
    eps = 10 ** -6

    for k, v in sorted(dict_clusters.items()):
        for clust in v:
            t0, t1, n = clust
            if t0 == t1:
                nodes_to_time_to_value[n][t0] = k
            else:
                nodes_to_time_to_value[n][t0 + eps] = k
                nodes_to_time_to_value[n][t1 - eps] = k

    # If node aren't present we input -1.

    for n, np in zip(S.nodes, S.node_presence):
        np = [S.times[0]] + np + [S.times[1]]
        for t0, t1 in zip(np[::2], np[1::2]):
            if t0 != t1:
                nodes_to_time_to_value[n][t0 + eps] = -1
                nodes_to_time_to_value[n][t1 - eps] = -1
    node_to_series = {}

    for n in nodes_to_time_to_value:
        if datetime:
            dates = [dt.datetime.fromtimestamp(ts) for ts in nodes_to_time_to_value[n]]
            Index = pd.to_datetime(dates)
        else:
            Index = nodes_to_time_to_value[n]
        node_to_series[n] = pd.Series(list(nodes_to_time_to_value[n].values()), index=Index)
        if withnan:
            node_to_series[n].replace(-1, numpy.nan, inplace=True)
        node_to_series[n].sort_index(inplace=True)
    return node_to_series


def read_stream_graph(path_links, path_nodes=None, node_label=True,
                      path_weights=None, path_trips=None
                      ):
    """
    tb : time of arrival (b: begin)
    te : time of departure (e: end)
    Input format :
    node file:
    id_node1 tb_0 te_0 tb_1 te_1 ... tb_n1 te_n1
    id_node2 tb_0 te_0 ...
    ...
    link file:
    id_node1 id_node2 tb_0 te_0 tb_1 te_1 ... tb_l1 te_l1
    id_node3 id_node4 tb_0 te_0 tb_1 te_1 ... tb_l2 te_l2
    ...

    :param path_trips:
    :param path_weights:
    :param node_label:
    :param path_links: path to store nodes and their time of presence
    :param path_nodes: path to store links and their time of presence
    :return: 
    """
    nodes = []
    node_presence = []
    links = []
    link_presence = []
    id_to_label, label_to_id = None, None
    alpha = -numpy.Infinity
    omega = numpy.Infinity

    if node_label:
        id_to_label = {}
        label_to_id = defaultdict(lambda: len(label_to_id))

    if path_nodes is not None:
        with open(path_nodes, 'r') as file_input:
            line_number = 0
            for line in file_input:
                ordinary_line = True
                line_copy = line[:]
                if line_number == 0 or line_number == 1:
                    line_copy = line_copy.strip().split("=")
                    if line_copy[0].strip()=="alpha":
                        ordinary_line = False
                        alpha = float(line_copy[1])
                    elif line_copy[0].strip()=="omega":
                        ordinary_line = False
                        omega = float(line_copy[1])
                line_number+=1


                line = line.strip().split(" ")
                if len(line) > 1 and ordinary_line:
                    if node_label:
                        n_label = str(line[0])
                        n = label_to_id[n_label]
                        id_to_label[n] = n_label
                    else:
                        n = int(line[0])
                    nodes.append(n)
                    np = [float(t) for t in line[1:]]
                    node_presence.append(np)
    with open(path_links, 'r') as file_input:
        for line in file_input:
            line = line.strip().split(" ")
            if len(line) > 2:
                if node_label:
                    u_label = str(line[0])
                    v_label = str(line[1])
                    if u_label not in label_to_id or v_label not in label_to_id:
                        #  Probably an empty node...
                        continue
                    # assert u_label in label_to_id
                    # assert v_label in label_to_id
                    u = label_to_id[u_label]
                    v = label_to_id[v_label]
                else:
                    u = int(line[0])
                    v = int(line[1])
                links.append((u, v))
                if path_nodes is None:
                    if u not in nodes:
                        nodes.append(u)
                    if v not in nodes:
                        nodes.append(v)
                lp = [float(t) for t in line[2:]]
                link_presence.append(lp)

    weights = []
    if path_weights:
        with open(path_weights, 'r') as file_input:
            for line in file_input:
                line = line.strip().split(" ")
                if len(line) > 2:
                    w = [float(t) for t in line[2:]]
                    weights.append(w)
    trips = []
    if path_trips:
        with open(path_trips, 'r') as file_input:
            for line in file_input:
                line = line.strip().split(" ")
                if len(line) > 2:
                    tp = [float(t) for t in line[2:]]
                    trips.append(tp)

    S = StreamGraph(times=[min([t for n in node_presence for t in n]),
                           max([t for n in node_presence for t in n])],
                    nodes=nodes,
                    node_presence=node_presence,
                    links=links,
                    link_presence=link_presence,
                    node_to_label=(id_to_label if id_to_label else None),
                    weights=(weights if weights else None),
                    trips=(trips if trips else None),
                    points = [],
                    alpha = alpha,
                    omega = omega)
    return S


def sum_presence(np):
    return sum([t1 - t0 for t0, t1 in zip(np[::2], np[1::2])])


def algo_kcores_batagelj(a_l, degrees):
    """
    Compute k_cores of a static graph from its adjacency list and nodes degrees
    References
    ----------
    [1] An O(m) Algorithm for Cores Decomposition of Networks
    Vladimir Batagelj and Matjaz Zaversnik, 2003.
    http://arxiv.org/abs/cs.DS/0310049

    :param a_l:
    :param degrees:
    :return:
    """
    bin = [0]
    sorted_nodes = sorted(degrees, key=degrees.get)
    curr_degree = 0
    for i, v in enumerate(sorted_nodes):
        if degrees[v] > curr_degree:
            bin.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(sorted_nodes)}
    cores = degrees
    for v in sorted_nodes:
        for w in a_l[v]:
            if cores[w] > cores[v]:
                a_l[w].discard(v)
                pos = node_pos[w]
                bin_start = bin[cores[w]]
                node_pos[w] = bin_start
                node_pos[sorted_nodes[bin_start]] = pos
                sorted_nodes[bin_start], sorted_nodes[pos] = sorted_nodes[pos], sorted_nodes[bin_start]
                bin[cores[w]] += 1
                cores[w] -= 1
    return cores


def read_stream_graph_from_json(path_nodes, path_links):
    """
    Parse a ".json" file and return a stream graph.
    
    :param path_nodes:
    :param path_links:
    :return:
    """
    nodes = []
    node_presence = []
    links = []
    link_presence = []
    nodes_to_id = {}
    id_to_node = {}
    nodes_to_label = {}
    with open(path_nodes, 'r') as file_input:
        nodes_json = json.load(file_input)
        for js in nodes_json["nodes"]:
            n_id = int(js["id"])
            if 'label' in js:
                n_label = str(js["label"])
            else:
                n_label = n_id
            n = len(nodes)
            nodes_to_id[n] = n_id
            id_to_node[n_id] = n
            nodes_to_label[n] = n_label
            nodes.append(n)
            np = []
            for i in js["intervals"]:
                np += [i["t0"], i["t1"]]
            np = sorted(np)
            # Remove Duplicatas
            id_to_remove = set()
            for i in range(len(np) - 1):
                if np[i] == np[i + 1]:
                    id_to_remove.add(i)
                    id_to_remove.add(i + 1)
            np = [np[i] for i in range(len(np)) if i not in id_to_remove]
            node_presence.append(np)
        times = nodes_json["timeExtent"]

    with open(path_links, 'r') as file_input:
        links_json = json.load(file_input)
        for js in links_json["links"]:
            u_id, v_id = int(js['node1']), int(js['node2'])
            links.append((id_to_node[u_id], id_to_node[v_id]))
            lp = []
            for i in js["intervals"]:
                lp += [i["t0"], i["t1"]]
            link_presence.append(lp)

    S = StreamGraph(times=times,
                    nodes=nodes,
                    node_presence=node_presence,
                    links=links,
                    link_presence=link_presence,
                    node_to_label=nodes_to_label,
                    node_to_id=nodes_to_id)
    return S


class StreamGraph:
    """
    A stream graph is a tool tailored to analyse and model sequences of links or, continuous sequences of graphs.

    """

    def __init__(self,
                 id=None,
                 times=None,
                 nodes=None,
                 node_to_label=None,
                 node_to_id=None,
                 node_presence=None,
                 links=None,
                 link_presence=None,
                 weights=None,
                 trips=None,
                 points=[],
                 alpha = -numpy.Infinity,
                 omega = numpy.Infinity
    ):
        """
        A basic constructor for a ``StreamGraph`` object
        
        :param id: A parameter to identify a stream graph.
        :param times: Continous interval of time during the stream graph exists 
        :param nodes: A list of nodes present in  the stream graph
        :param node_presence : List of lists in the same order as the nodes. Each list contains 
        succescively the time of apparition and time of disparition of the node.
        :param links : A list of links present in the stream graph
        :param link_presence : same as node_presence
        """
        self.id = id
        self.times = times
        self.nodes = nodes
        self.node_to_label = node_to_label
        self.node_to_id = node_to_id
        self.node_presence = node_presence
        self.links = links
        self.link_presence = link_presence
        self.weights = weights
        self.trips = trips
        self.points = points
        l = self.event_times()
        if alpha == -numpy.Infinity:
            self.alpha = min(l)
        else:
            self.alpha = alpha
        if omega == numpy.Infinity:
            self.omega = max(l)
        else:
            self.omega = omega

    def check_integrity(self):
        """
        Check node presence and link presence for overlapping time windows
        Check that a link presence include both node presence

        :return: True if the structure is coherent or an error with the problematic link/node.
        """
        # Easily optimisable, suffit de comparer les éléments de liste pairwise puis de double check if TRUE
        for n, np in zip(self.links, self.link_presence):
            assert list(sorted(np)) == np
            for t0, t1 in zip(np[::2], np[1::2]):
                for t2, t3 in zip(np[::2], np[1::2]):
                    if (t0, t1) != (t2, t3):
                        if t0 <= t3 and t2 <= t1:
                            raise ValueError("Integrity compromised on node : "
                                             + str(n) +
                                             ". Check node presence (probably overlapping intervals) !\n")

        for l, lp in zip(self.links, self.link_presence):
            assert list(sorted(lp)) == lp
            id1 = self.nodes.index(l[0])
            id2 = self.nodes.index(l[1])
            for lt0, lt1 in zip(lp[::2], lp[1::2]):
                check_1 = False
                check_2 = False
                for nt0, nt1 in zip(self.node_presence[id1][::2],
                                    self.node_presence[id1][1::2]):
                    if nt0 <= lt0 and nt1 >= lt1:
                        check_1 = True
                        continue
                for nt0, nt1 in zip(self.node_presence[id2][::2],
                                    self.node_presence[id2][1::2]):
                    if nt0 <= lt0 and nt1 >= lt1:
                        check_2 = True
                        continue
                if not check_1 or not check_2:
                    raise ValueError("Integrity compromised on link : "
                                     + str(l) + " at time :" + str((lt0, lt1)) +
                                     ". Check node presence !\n" + "Node " + str(l[0]) +
                                     " presence:" + str(self.node_presence[l[0]]) + "\nNode " + str(l[1]) +
                                     " presence:" + str(self.node_presence[l[1]]))
        print("Integrity check ok !")
        return True

    def add_node(self, node, node_presence):
        """
        Add a node to the stream graph, if not already present :
        this new node will be represented by a an integer, corresponding to the length of self.nodes

        :param node: Label of the node to add
        :param node_presence: Presence times of the added node [b,e,b',e',...]
        :return: The id corresponding to the added node
        """
        if self.node_to_label is not None:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            if node in label_to_node:
                new_node = label_to_node[node]
                assert self.node_presence[new_node][-1] < node_presence[0]
                self.node_presence[new_node] += node_presence
            else:
                new_node = len(self.nodes)
                self.node_to_label[new_node] = node
                self.nodes.append(new_node)
                self.node_presence.append(node_presence)
        else:
            if node < len(self.nodes):
                new_node = node
                assert self.node_presence[node][-1] < node_presence[0]
                self.node_presence[node] += node_presence
            else:
                new_node = len(self.nodes)
                self.nodes.append(new_node)
                self.node_presence.append(node_presence)
        return new_node

    def add_nodes(self, nodes_list, node_presence_list):
        for n, np in zip(nodes_list, node_presence_list):
            self.add_node(n, np)

    def add_link(self, link, link_presence):
        """
        Add a new link to the stream graph. If the extremities are not already present we add them to the stream graph.
        Their presence times will be the one the link.

        :param link: (A,B) where 'A' and 'B' are the nodes' labels
        :param link_presence: Presence times of the added link [b,e,b',e',...]
        :return:
        """
        u, v = link
        if self.node_to_label is not None:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            if u in label_to_node:
                new_u = label_to_node[u]
            else:
                new_u = self.add_node(u, link_presence)
            if v in label_to_node:
                new_v = label_to_node[v]
            else:
                new_v = self.add_node(v, link_presence)
        else:
            if u in self.nodes:
                new_u = u
            else:
                new_u = self.add_node(u, link_presence)
            if v in self.nodes:
                new_v = v
            else:
                new_v = self.add_node(v, link_presence)

        if (new_u, new_v) in self.links:
            assert self.link_presence[self.links.index((new_u, new_v))][-1] < link_presence[0]
            self.link_presence[self.links.index((new_u, new_v))] += link_presence
        elif (new_v, new_u) in self.links:
            assert self.link_presence[self.links.index((new_v, new_u))][-1] < link_presence[0]
            self.link_presence[self.links.index((new_v, new_u))] += link_presence
        else:
            new_link = (new_u, new_v)
            self.links.append(new_link)
            self.link_presence.append(link_presence)

    def add_links(self, links_list, link_presence_list):
        for l, lp in zip(links_list, link_presence_list):
            self.add_link(l, lp)

    #####################################
    #       Events Representation       #
    #####################################

    def event_times(self, all=False):
        """
        Return distinct event times in the Stream Graph

        :param all:
        :return:
        """
        if all:
            event_times = sorted([t for np in self.node_presence for t in np] +
                                 [t for lp in self.link_presence for t in lp] +
                                 [t  for t in self.points]
            )
            return event_times
        else:
            event_t_nodes = set([t for np in self.node_presence for t in np])
            event_t_links = set([t for lp in self.link_presence for t in lp])
            event_t_points = set([t  for t in self.points])
            return event_t_links | event_t_nodes | event_t_points

    def number_of_event_times(self):
        """
        Return the number of distinct event times in the Stream Graph

        :return:
        """
        return len(self.event_times())

    def ordered_events(self, weights_or_trips=False):
        """
        Return an ordered of all the events (nodes and links arrivals or departure) occuring in the stream graph.

        :param weights_or_trips:
        :return:
        """
        links = []
        if weights_or_trips:
            if self.weights and self.trips:
                for l, lp, we, tr in zip(self.links, self.link_presence, self.weights, self.trips):
                    for t0, t1, w, d in zip(lp[::2], lp[1::2], we, tr):
                        u, v = l
                        links.append((1, t0, t1, u, v, w, d))  # code each link, 1 for a beginning, -1 for an ending
                        links.append((-1, t1, u, v, w, d))

            elif self.weights:
                for l, lp, we in zip(self.links, self.link_presence, self.weights):
                    for t0, t1, w in zip(lp[::2], lp[1::2], we):
                        u, v = l
                        links.append((1, t0, t1, u, v, w, 0))  # code each link, 1 for a beginning, -1 for an ending
                        links.append((-1, t1, u, v, w, 0))

            elif self.trips:
                for l, lp, tr in zip(self.links, self.link_presence, self.trips):
                    for t0, t1, d in zip(lp[::2], lp[1::2], tr):
                        u, v = l
                        links.append((1, t0, t1, u, v, 1, d))  # code each link, 1 for a beginning, -1 for an ending
                        links.append((-1, t1, u, v, 1, d))
            else:
                for l, lp in zip(self.links, self.link_presence):
                    for t0, t1 in zip(lp[::2], lp[1::2]):
                        u, v = l
                        links.append((1, t0, t1, u, v, 1, 0))  # code each link, 1 for a beginning, -1 for an ending
                        links.append((-1, t1, u, v, 1, 0))
        else:
            for l, lp in zip(self.links, self.link_presence):
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    u, v = l
                    links.append((1, t0, t1, u, v))  # code each link, 1 for a beginning, -1 for an ending
                    links.append((-1, t1, u, v))
        nodes = []
        for n, np in zip(self.nodes, self.node_presence):
            for t0, t1 in zip(np[::2], np[1::2]):
                nodes.append((2, t0, t1, n))  # code a node arrival with a 2
                nodes.append((-2, t1, n))  # code a node departure with a -2
        events = sorted(links + nodes, key=lambda x: (x[1], -x[0]))
        return events

    def ordered_batch_events(self):
        """
        Return an ordered of batch events (event happening at the same time).
        In each batch there can be (in the same order): node arrival, link arrival, link departure, node departure.

        :return:
        """

        batchs = []
        E = self.ordered_events()
        t_old = E[0][1]
        c_old = E[0][0]
        current_batch = []
        for e in E:
            c, t = e[0], e[1]
            if t == t_old and c == c_old:
                current_batch.append(e)
            else:
                batchs.append(current_batch)
                current_batch = [e]
                t_old = t
                c_old = c

        if current_batch:
            batchs.append(current_batch)
        return batchs

    #######################################
    #       Arrivals Events               #
    #######################################

    def ordered_arrivals(self, free_memory=False):
        """
        Return an ordered list of arrival events (node or link arrival) occuring in the ``StreamGraph``.

        :return:
        """

        nodes = []
        for n, np in zip(self.nodes, self.node_presence):
            for t0, t1 in zip(np[::2], np[1::2]):
                nodes.append((2, t0, t1, n))  #  code a node arrival with a 2
        if free_memory:
            self.nodes = []
            self.node_presence = []

        links = []
        for l, lp in zip(self.links, self.link_presence):
            for t0, t1 in zip(lp[::2], lp[1::2]):
                u, v = l
                links.append((1, t0, t1, u, v))  # code each link, 1 for a beginning, -1 for an ending

        if free_memory:
            self.links = []
            self.link_presence = []

        events = sorted(links + nodes, key=lambda x: (x[1], -x[0]))
        return events

    #####################################
    #       Links Representation        #
    #####################################

    def augmented_ordered_links(self, weights_or_trips=False):
        """


        :return:
        """
        links = []
        if weights_or_trips:
            if self.weights and self.trips:
                for l, lp, we, tr in zip(self.links, self.link_presence, self.weights, self.trips):
                    for t0, t1, w, d in zip(lp[::2], lp[1::2], we, tr):
                        nu, nv = self.get_according_node_presence_from_link(l, t0, t1)
                        if nu and nv:
                            links.append(
                                (1, t0, t1, nu, nv, w, d))  # code each link, 1 for a beginning, -1 for an ending
                            links.append((-1, t1, nu, nv, w, d))

            elif self.weights:
                for l, lp, we in zip(self.links, self.link_presence, self.weights):
                    for t0, t1, w in zip(lp[::2], lp[1::2], we):
                        nu, nv = self.get_according_node_presence_from_link(l, t0, t1)
                        if nu and nv:
                            links.append(
                                (1, t0, t1, nu, nv, w, 0))  # code each link, 1 for a beginning, -1 for an ending
                            links.append((-1, t1, nu, nv, w, 0))

            elif self.trips:
                for l, lp, tr in zip(self.links, self.link_presence, self.trips):
                    for t0, t1, d in zip(lp[::2], lp[1::2], tr):
                        nu, nv = self.get_according_node_presence_from_link(l, t0, t1)
                        if nu and nv:
                            links.append(
                                (1, t0, t1, nu, nv, 1, d))  # code each link, 1 for a beginning, -1 for an ending
                            links.append((-1, t1, nu, nv, 1, d))
            else:
                for l, lp in zip(self.links, self.link_presence):
                    for t0, t1 in zip(lp[::2], lp[1::2]):
                        nu, nv = self.get_according_node_presence_from_link(l, t0, t1)
                        if nu and nv:
                            links.append(
                                (1, t0, t1, nu, nv, 1, 0))  # code each link, 1 for a beginning, -1 for an ending
                            links.append((-1, t1, nu, nv, 1, 0))
        else:
            for l, lp in zip(self.links, self.link_presence):
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    nu, nv = self.get_according_node_presence_from_link(l, t0, t1)
                    if nu and nv:
                        links.append((1, t0, t1, nu, nv))  # code each link, 1 for a beginning, -1 for an ending
                        links.append((-1, t1, nu, nv))
        links = sorted(links, key=lambda x: (x[1], -x[0]))
        return links

    def ordered_batch_links(self, free_memory=False):
        links = []
        batchs = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            for t0, t1 in zip(lp[::2], lp[1::2]):
                links.append((1, t0, t1, u, v))  # code each link, 1 for a beginning, -1 for an ending
                links.append((-1, t1, u, v))
        if free_memory:
            self.nodes = []
            self.node_presence = []
            self.links = []
            self.link_presence = []

        links = sorted(links, key=lambda x: (x[1], -x[0]))
        t_old = links[0][1]
        c_old = links[0][0]
        current_batch = []
        for e in links:
            c, t = e[0], e[1]
            if t == t_old and c == c_old:
                current_batch.append(e)
            else:
                batchs.append(current_batch)
                current_batch = [e]
                t_old = t
                c_old = c
        if current_batch:
            batchs.append(current_batch)
        return batchs

    def ordered_links(self):
        """
        :return:
        """
        links = []
        if self.weights and self.trips:
            for l, lp, we, tr in zip(self.links, self.link_presence, self.weights, self.trips):
                u, v = l
                for t0, t1, w, d in zip(lp[::2], lp[1::2], we, tr):
                    links.append((1, t0, t1, u, v, w, d))  # code each link, 1 for a beginning, -1 for an ending
                    links.append((-1, t1, u, v, w, d))
            links = sorted(links, key=lambda x: (x[1], -x[0]))

        elif self.weights:
            for l, lp, we in zip(self.links, self.link_presence, self.weights):
                u, v = l
                for t0, t1, w in zip(lp[::2], lp[1::2], we):
                    links.append((1, t0, t1, u, v, w))  # code each link, 1 for a beginning, -1 for an ending
                    links.append((-1, t1, u, v, w))
            links = sorted(links, key=lambda x: (x[1], -x[0]))

        elif self.trips:
            for l, lp, tr in zip(self.links, self.link_presence, self.trips):
                u, v = l
                for t0, t1, d in zip(lp[::2], lp[1::2], tr):
                    links.append((1, t0, t1, u, v, d))  # code each link, 1 for a beginning, -1 for an ending
                    links.append((-1, t1, u, v, d))
            links = sorted(links, key=lambda x: (x[1], -x[0]))
        else:
            for l, lp in zip(self.links, self.link_presence):
                u, v = l
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    links.append((1, t0, t1, u, v))  # code each link, 1 for a beginning, -1 for an ending
                    links.append((-1, t1, u, v))
            links = sorted(links, key=lambda x: (x[1], -x[0]))
        return links

    ####################################################################
    #               WRITERS                                            #
    ####################################################################

    def sg_to_streamfig(self, output_file, language):
        """
        tb : time of arrival (b: begin)
        te : time of departure (e: end)
        Output format :
        a preciser

        :param output_file: path to store nodes, links and their time of presence
        :return:
        """


        if self.node_to_label:
            with open(output_file + '.py', 'w') as file_output:
                file_output.write("import streamfig\n")
                file_output.write("s = streamfig.StreamFig(alpha="+ str(self.alpha) +", omega=" + str(self.omega) + ", directed = True)\n")
                for n, np in zip(self.nodes, self.node_presence):
                    file_output.write("s.addNode(\"" +str(self.node_to_label[n]) + "\",[ ")
                    for i in range(0,len(np)-2,2):
                        file_output.write( str((np[i],np[i+1]))+", " )
                    i = len(np)-2
                    file_output.write( str((np[i],np[i+1]))+"]) " )
                    file_output.write("\n")

                for j in range(0,len(self.links)):
                    for i in range(0,len(self.link_presence[j]),2 ):
                        file_output.write("s.addLink(\"" + str(self.node_to_label[self.links[j][0]]) + "\", \"" + str(self.node_to_label[self.links[j][1]]) + "\", " + str(self.link_presence[j][i]) + ","  + str(self.link_presence[j][i+1]) + ")\n" )
                    file_output.write("\n")
                file_output.write("s.addTimeLine(ticks=2) \n")
                file_output.write("s.save_make_figure(\""+language+"\",0) \n")

        else:
            with open(output_file + '.py', 'w') as file_output:
                file_output.write("import streamfig\n")
                file_output.write("s = streamfig.StreamFig(alpha="+ self.alpha +", omega=" + self.omega + ", directed = True)\n")
                for n, np in zip(self.nodes, self.node_presence):
                    file_output.write("s.addNode(\"" +str(n) + "\",[ ")
                    for i in range(0,len(np)-2,2):
                        file_output.write( str((np[i],np[i+1]))+", " )
                    i = len(np)-2
                    file_output.write( str((np[i],np[i+1]))+"]) " )
                    file_output.write("\n")

                for j in range(0,self.links):
                    for i in range(0,len(self.link_presence[l]),2 ):
                        file_output.write("s.addLink(\"" + str(self.node_to_label[self.links[j][0]]) + "\", \"" + str(self.node_to_label[self.links[j][1]]) + "\", " + str(self.link_presence[j][i]) + ","  + str(self.link_presence[j][i+1]) + ")\n" )
                    file_output.write("\n")
                file_output.write("s.addTimeLine(ticks=2) \n")
                file_output.write("s.save_make_figure(\""+language+"\",0) \n")


    def write_to_sg(self, output_file):
        """
        tb : time of arrival (b: begin)
        te : time of departure (e: end)
        Output format :
        node file:
        id_node1 tb_0 te_0 tb_1 te_1 ... tb_n1 te_n1
        id_node2 tb_0 te_0 ...
        ...
        link file:
        id_node1 id_node2 tb_0 te_0 tb_1 te_1 ... tb_l1 te_l1
        id_node3 id_node4 tb_0 te_0 tb_1 te_1 ... tb_l2 te_l2
        ...

        :param output_file: path to store nodes, links and their time of presence
        :return:
        """


        if self.node_to_label:
            with open(output_file + '_nodes.sg', 'w') as file_output:
                if self.alpha != -numpy.Infinity:
                    file_output.write("alpha =" + self.alpha +" \n")
                if self.omega != -numpy.Infinity:
                    file_output.write("omega =" + self.omega +" \n")

                for n, np in zip(self.nodes, self.node_presence):
                    file_output.write(str(self.node_to_label[n]) + " ")
                    for t in np:
                        file_output.write(str(t) + " ")
                    file_output.write("\n")
            with open(output_file + '_links.sg', 'w') as file_output:
                for l, lp in zip(self.links, self.link_presence):
                    file_output.write(str(self.node_to_label[l[0]]) + " " + str(self.node_to_label[l[1]]) + " ")
                    for t in lp:
                        file_output.write(str(t) + " ")
                    file_output.write("\n")
        else:
            with open(output_file + '_nodes.sg', 'w') as file_output:
                for n, np in zip(self.nodes, self.node_presence):
                    file_output.write(str(n) + " ")
                    for t in np:
                        file_output.write(str(t) + " ")
                    file_output.write("\n")
            with open(output_file + '_links.sg', 'w') as file_output:
                for l, lp in zip(self.links, self.link_presence):
                    file_output.write(str(l[0]) + " " + str(l[1]) + " ")
                    for t in lp:
                        file_output.write(str(t) + " ")
                    file_output.write("\n")

    def write_to_csv(self, output_name):
        """
        Write the stream graph to CSV format(node1;node2;start_time;duration).

        :param output_name:
        :return:
        """
        links = []
        if self.node_to_label:
            for l, lp in zip(self.links, self.link_presence):
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    links.append((t0, t1 - t0, self.node_to_label[l[0]],
                                  self.node_to_label[l[1]],))
        else:
            for l, lp in zip(self.links, self.link_presence):
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    links.append((t0, t1 - t0, l[0], l[1],))
        links = sorted(links, key=lambda x: (x[0], x[1]))
        with open(output_name, 'w', newline='') as file_output:
            stream_writer = csv.writer(file_output, delimiter=';')
            for l in links:
                stream_writer.writerow(l)

    def write_to_lsf(self, output_name):
        links = []
        if self.node_to_label:
            for l, lp in zip(self.links, self.link_presence):
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    links.append((t0, t1, self.node_to_label[l[0]],
                                  self.node_to_label[l[1]]))
        else:
            for l, lp in zip(self.links, self.link_presence):
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    links.append((t0, t1, l[0], l[1]))
        links = sorted(links, key=lambda x: (x[2], x[3]))
        links = [str(l[0]) + " " + str(l[1]) + " " + str(l[2]) + " " + str(l[3])
                 for l in links]
        with open(output_name + ".lsf", 'w', newline='') as otp:
            otp.write("alpha " + str(self.times[0]) + "\n")
            otp.write("omega " + str(self.times[1]) + "\n")
            for l in links:
                otp.write(l + "\n")

    def write_to_json(self, output_name, index_pos=None):
        """
        Write to a JSON format. For the tool stream-graph-visualisation.

        :param output_name: Name prefix of json files. Will be stored under "output_name_node_activity.json"\
        and "output_name_link_presence.json"
        :param index_pos:
        :return:
        """
        nodes_json = self.node_activity_to_json(index_pos)
        links_json = self.link_presence_to_json()
        with open(output_name + "_node_activity.json", 'w') as file_output:
            json.dump(nodes_json, file_output)
        with open(output_name + "_link_presence.json", 'w') as file_output:
            json.dump(links_json, file_output)

    def node_activity_to_json(self, node_to_position=None, sort_type=None):
        """
        For the tool stream-graph-visualisation.

        :return: node activity in JSON
        """

        if self.node_to_id is None:
            self.node_to_id = {n: n for n in self.nodes}
        degrees_partition = self.degrees_partition()
        isolated_nodes = self.isolated_nodes()
        degrees_partition[0] = isolated_nodes
        # print("Degrees partition :",degrees_partition)
        if node_to_position is None:
            if self.node_to_id:
                node_to_position = {self.node_to_id[n]: n for n in self.nodes}
            else:
                node_to_position = {n: n for n in self.nodes}
        if sort_type == "arrival":
            node_to_position = self.node_position_by_arrival()
        elif sort_type == "increasing_degree":
            node_to_position = self.node_position_by_increasing_degree(degrees_partition)
        elif sort_type == "peak_degree_arrival":
            node_to_position = self.node_position_by_peak_degree_arrival(degrees_partition)
        # print("SORT TYPE :",sort_type)
        # print(" NODE TO POSITION :",node_to_position)
        max_d = 0
        min_d = 0
        nodes_json = {}
        if self.node_to_label:
            nodes_json["nodes"] = [{"id": (self.node_to_id[n] if self.node_to_id else n),
                                    "label": str(self.node_to_label[n]),
                                    "pos": node_to_position[(self.node_to_id[n] if self.node_to_id else n)],
                                    "intervals": []}
                                   for n in self.nodes]
            for degree, nodes_set in degrees_partition.items():
                for n in nodes_set:
                    t0, t1, u = n
                    if u in self.nodes:  # important pour la visualisation si le noeud est dans les liens mais
                        # pas la liste des noeuds
                        max_d = max(degree, max_d)
                        min_d = min(degree, min_d)
                        nodes_json["nodes"][u]["intervals"].append({"t0": t0, "t1": t1, "degree": degree})
        else:
            nodes_json["nodes"] = [{"id": (self.node_to_id[n] if self.node_to_id else n),
                                    "label": n,
                                    "pos": node_to_position[(self.node_to_id[n] if self.node_to_id else n)],
                                    "intervals": []}
                                   for n in self.nodes]
            for degree, nodes_set in degrees_partition.items():
                for n in nodes_set:
                    t0, t1, u = n
                    nodes_json["nodes"][u]["intervals"].append({"t0": t0, "t1": t1, "degree": degree})

        nodes_json["timeExtent"] = [self.times[0], self.times[1]]
        nodes_json["degreeExtent"] = [min_d, max_d]
        return nodes_json

    def link_presence_to_json(self, link_to_position=None, sort_type=None):
        """
        For the tool stream-graph-visualisation.

        :return: link presence in JSON
        """
        links_json = {"links": [], "timeExtent": [self.times[0], self.times[1]]}
        max_duration_link = 0
        min_duration_link = self.times[1] - self.times[0]

        if link_to_position is None:
            if self.node_to_id:
                link_to_position = {(self.node_to_id[l[0]], self.node_to_id[l[1]]): i for i, l in
                                    enumerate(self.links)}
            else:
                link_to_position = {l: i for i, l in
                                    enumerate(self.links)}
        if sort_type == "duration":
            link_to_position = self.link_position_by_duration()
        elif sort_type == "arrival":
            link_to_position = self.link_position_by_arrival()
        # print("SORT TYPE :",sort_type)
        # print("LINK TO POSITION :",link_to_position)

        for l, lp in zip(self.links, self.link_presence):
            link_presence = []
            duration = 0
            for t0, t1 in zip(lp[::2], lp[1::2]):
                link_presence.append({'t0': t0, 't1': t1})
                duration += t1 - t0
            max_duration_link = max(max_duration_link, duration)
            min_duration_link = min(min_duration_link, duration)
            links_json["links"].append({"node1": (str(self.node_to_id[l[0]]) if self.node_to_id else l[0]),
                                        "node2": (str(self.node_to_id[l[1]]) if self.node_to_id else l[1]),
                                        "intervals": link_presence,
                                        "duration": duration,
                                        "pos": (link_to_position[(self.node_to_id[l[0]], self.node_to_id[l[1]])]
                                                if self.node_to_id else link_to_position[l])})

        links_json["durationExtent"] = [min_duration_link, max_duration_link]
        return links_json

    def describe(self):
        """
        Print a global description of a ``StreamGraph`` object in terms of number of nodes and links.

        :return: None
        """
        print("Nb of Nodes : ", len(self.nodes))
        print("Nb of segmented nodes : ", sum([len(item) / 2 for item in self.node_presence]))
        print("Nb of links : ", len(self.links))
        print("Nb of segmented links : ", sum([len(item) / 2 for item in self.link_presence]))
        print("Nb of event times : ", self.number_of_event_times())

    ####################################################################
    #               Static Graph Properties                            #
    ####################################################################

    def graph_property(self, node_property_function_nx,
                       node_list=None,
                       format="cluster",
                       stable_components=None,
                       postprocess=True,
                       n_jobs=-1,
                       datetime=True,
                       withnan=True,
                       eps=None):

        if stable_components is None:
            stable_components = self.stable_connected_components(format="object_with_links",
                                                                 node_list=node_list)
        if format != "cluster" and format != "signal":
            raise KeyError("Read the API documentation #USERISLIMITED")

        if format == "signal":
            return self.property_to_signal(node_property_function_nx, datetime, eps, n_jobs, postprocess,
                                           stable_components, withnan, node_list)

        if format == "cluster":
            return self.property_to_cluster(node_property_function_nx, n_jobs, postprocess, stable_components,
                                            node_list)

    def property_to_cluster(self, node_property_function_nx, n_jobs, postprocess, stable_components, node_list):
        def para_cluster(sc, node_list):
            prop_to_clusters = defaultdict(list)
            a_l = sc.to_adjacency_list()
            if a_l:
                G = nx.from_dict_of_lists(a_l)
                prop = node_property_function_nx(G)
                for n in prop:
                    if node_list is None:
                        prop_to_clusters[prop[n]].append((sc.times[0], sc.times[1], n))
                    elif n in node_list:
                        prop_to_clusters[prop[n]].append((sc.times[0], sc.times[1], n))
            else:  #  Isolated nodes (default value 0)
                for n in sc.nodes:
                    if node_list is None:
                        prop_to_clusters[0].append((sc.times[0], sc.times[1], n))
                    elif n in node_list:
                        prop_to_clusters[0].append((sc.times[0], sc.times[1], n))
            return prop_to_clusters

        r = Parallel(n_jobs=n_jobs)(delayed(para_cluster)(sc, node_list) for sc in stable_components)
        prop_to_clusters = defaultdict(list)
        for d in r:
            for k, v in d.items():
                prop_to_clusters[k] += v
        if postprocess is True:
            prop_to_clusters = postprocess_clusters(prop_to_clusters)
        return prop_to_clusters

    def property_to_signal(self, node_property_function_nx, datetime, eps,
                           n_jobs, postprocess, stable_components, withnan,
                           node_list):
        if eps is None:
            eps = (self.times[1] - self.times[0]) / (self.number_of_event_times() * 1000)

        def para_signal(sc, node_list):
            node_to_signals = defaultdict(dict)
            a_l = sc.to_adjacency_list()
            t0, t1 = sc.times[0], sc.times[1]
            if t0 != t1:
                t0 = t0 + eps
                t1 = t1 - eps
            if a_l:
                G = nx.from_dict_of_lists(a_l)
                prop = node_property_function_nx(G)
                for n in prop:
                    if node_list is None:
                        node_to_signals[n][t0] = prop[n]
                        node_to_signals[n][t1] = prop[n]
                    elif n in node_list:
                        node_to_signals[n][t0] = prop[n]
                        node_to_signals[n][t1] = prop[n]
            else:  #  Isolated nodes (default value 0)
                for n in sc.nodes:
                    if node_list is None:
                        node_to_signals[n][t0] = 0
                        node_to_signals[n][t1] = 0
                    elif n in node_list:
                        node_to_signals[n][t0] = 0
                        node_to_signals[n][t1] = 0
            return node_to_signals

        r = Parallel(n_jobs=n_jobs)(delayed(para_signal)(sc, node_list) for sc in stable_components)
        node_to_time_to_value = defaultdict(dict)
        for n_to_sig in r:
            for n in n_to_sig:
                node_to_time_to_value[n].update(n_to_sig[n])
        if node_list is None:
            for n in self.nodes:
                np = self.node_presence[n]
                np = [self.times[0]] + np + [self.times[1]]
                for t0, t1 in zip(np[::2], np[1::2]):
                    if t0 != t1:
                        node_to_time_to_value[n][t0 + eps] = -1
                        node_to_time_to_value[n][t1 - eps] = -1
        else:
            for n in node_list:
                np = self.node_presence[n]
                np = [self.times[0]] + np + [self.times[1]]
                for t0, t1 in zip(np[::2], np[1::2]):
                    if t0 != t1:
                        node_to_time_to_value[n][t0 + eps] = -1
                        node_to_time_to_value[n][t1 - eps] = -1
        if postprocess:
            node_to_series = {}
            for n in node_to_time_to_value:
                if datetime:
                    dates = [dt.datetime.fromtimestamp(ts) for ts in node_to_time_to_value[n]]
                    Index = pd.to_datetime(dates)
                else:
                    Index = node_to_time_to_value[n]
                node_to_series[n] = pd.Series(list(node_to_time_to_value[n].values()), index=Index)
                if withnan:
                    node_to_series[n].replace(-1, numpy.nan, inplace=True)
                node_to_series[n].sort_index(inplace=True)
            return node_to_series
        return node_to_time_to_value

    def newman_communities(self, stable_dag=None, n_jobs=-1):
        from networkx.algorithms import community

        if stable_dag is None:
            stable_dag = self.stable_dag()

        def para_comp(sc):
            # print("comp times :", sc.times)
            # print("comp nodes :", sc.nodes)
            a_l = sc.to_adjacency_list()
            # clusters = []
            if a_l:
                G = nx.from_dict_of_lists(a_l)
                communities = next(community.girvan_newman(G))
                sc.clusters = [set(c) for c in communities]
                # for i,c in enumerate(communities):
                #     clusters.append([(sc.times[0], sc.times[1], n) for n in c])
                # print("clusters :",clusters)
            else:  #  Isolated nodes (default value 0)
                # for n in sc.nodes:
                #     clusters.append([(sc.times[0], sc.times[1], n)])
                sc.clusters = [{n} for n in sc.nodes]
            # return clusters
            return

        Parallel(n_jobs=n_jobs, backend="threading")(delayed(para_comp)(sc) for sc in stable_dag.c_nodes)
        # return [c for l in r for c in l]

    def louvain_communities(self, stable_dag=None, n_jobs=-1):
        import community as community_louvain

        if stable_dag is None:
            stable_dag = self.stable_dag()

        def para_comp(sc):
            # print("comp times :", sc.times)
            # print("comp nodes :", sc.nodes)
            a_l = sc.to_adjacency_list()
            # dict_clust = defaultdict(list)
            if a_l:
                G = nx.from_dict_of_lists(a_l)
                d_clusters = community_louvain.best_partition(G)
                # print("clusters louvain :",d_clusters)
                sc.clusters = [set() for _ in set(d_clusters.values())]
                for k, v in d_clusters.items():
                    sc.clusters[v].add(k)
                # for n in clusters:
                # dict_clust[clusters[n]].append((sc.times[0], sc.times[1], n))
            else:
                sc.clusters = [{n} for n in sc.nodes]
                # for n in sc.nodes:
                # dict_clust[-1].append((sc.times[0], sc.times[1], n))
            # return [v for v in dict_clust.values()]
            return

        Parallel(n_jobs=n_jobs, backend="threading")(delayed(para_comp)(sc) for sc in stable_dag.c_nodes)
        # return [c for l in r for c in l]

    def graph_communities(self, community_function='louvain', node_list=None,
                          n_jobs=-1, postprocess=True, sdag=None, alpha=0.7):
        """
        Compute communities in a Stream Graph according to a community function

        :param alpha:
        :param sdag:
        :param community_function:
        :param node_list:
        :param n_jobs:
        :param postprocess:
        :return:
        """

        if sdag is None:
            sdag = self.stable_dag(node_list=node_list)

        # if stable_components is None:
        #     stable_components = self.stable_connected_components(format="object_with_links",
        #                                                          node_list = node_list)

        if community_function == 'louvain':
            self.louvain_communities(sdag, n_jobs)

        elif community_function == 'newman':
            self.newman_communities(sdag, n_jobs)
        else:
            raise ValueError('Community function not supported.'
                             ' Currently supported functions are Louvain and Newman')
        if postprocess:
            return postprocess_communities(sdag, alpha=alpha)
        else:
            return [[(sc.times[0], sc.times[1], n) for n in c] for sc in sdag.c_nodes for c in sc.clusters]

    ####################################################################
    #               PLOT                                               #
    ####################################################################

    def plot(self, clusters=None, nodes_list=None, plot_links=True, title=None, fontsize=26,
             legend=True,
             timestamp=False, cmap=None, arrivals_marker=None,
             lw_nodes=3, lw_links=2.5, ax=None):
        """
        Display in an elegant way a (small) stream graph using ``matplotlib``.

        :param clusters: A list of clusters that can represents temporal nodes (a list of (t0,t1,u)) or
                        a dictionnary linking temporal nodes to a given value.
        :param nodes_list: A list of nodes to display, will ignore nodes not in this list.
        :param plot_links: A boolean value to plot temporal links or not
        :param title: An optional title
        :param fontsize:
        :param legend: A boolean value to display a legend, a colorbar for the current displayed property
                        (only available with the option clusters and ``clusters`` must be a dictionary)
        :param timestamp: A boolean value to convert the x-axis (the time) into human readable timstamp (dates).
        :param cmap: A colormap that will be used to display ``clusters``
        :param arrivals_marker: A boolean value to plot or not link arrivals with a marker
        :param lw_nodes: linewidth of temporal nodes
        :param lw_links: linewidth of temporal links
        :param ax: A ``matplotlib`` axe object
        :return: A ``matplotlib`` axe object
        """
        n_nodes = max(self.nodes)

        # Force ``nodes_list`` to be a set
        if nodes_list is not None:
            nodes_list = set(nodes_list)

        c_map = get_cmap(n_nodes + 2, cmap=cmap)
        nodes_cmap = [c_map(i) for i in self.nodes]

        if ax is None:
            fig, ax = plt.subplots()

        # Plot Links
        if plot_links:
            if arrivals_marker is None:
                if clusters or len(self.links) >= 25:
                    arrivals_marker = False
                else:
                    arrivals_marker = True

            self._plot_links(ax, lw_links=lw_links, arrivals_marker=arrivals_marker)

        # Plot clusters and nodes or only nodes
        list_legend = None
        if clusters:
            list_legend = self._plot_clusters(ax, clusters, cmap_clusters=cmap, nodes_list=nodes_list,
                                              lw_nodes=lw_nodes,
                                              legend=legend)
        else:
            self._plot_nodes(ax, nodes_cmap=nodes_cmap,
                             nodes_list=nodes_list,
                             lw_nodes=lw_nodes)

        # Set xticks
        xticks = numpy.linspace(int(self.times[0]), int(self.times[1]), 11)
        xticks = [numpy.around(e, decimals=2) for e in xticks]
        ax.set_xticks(xticks)
        if timestamp is True:

            dates = [dt.datetime.fromtimestamp(ts) for ts in xticks]
            xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            ax.xaxis.set_major_formatter(xfmt)
            ax.set_xticklabels(dates,
                               rotation=30, fontname='Garamond', fontsize=fontsize, color='#476b6b')
        else:
            ax.set_xticklabels(xticks,
                               fontname='Garamond', fontsize=fontsize, color='#476b6b')

        # Set yticks
        yticks = numpy.linspace(int(self.nodes[0]), int(self.nodes[-1]), min(len(self.nodes), 25), dtype=int)
        ax.set_yticks(yticks)

        if self.node_to_label:
            ax.set_yticklabels([self.node_to_label[i] for i in yticks], fontname='Garamond',
                               fontsize=fontsize,
                               color='#666699')
        else:
            ax.set_yticklabels([self.nodes[i] for i in yticks], fontname='Garamond', fontsize=fontsize,
                               color='#666699')

        # Set axes labels
        ax.set_ylabel("Nodes", fontname='Garamond', fontsize=fontsize, color='#666699')
        ax.set_xlabel("t", fontname='Garamond', fontsize=fontsize, color='#476b6b')

        # Set axes limits
        ax.set_xlim((self.times[0] - 0.3, self.times[1] + 0.3))
        ax.set_ylim((-0.3, max(self.nodes) + 0.3))

        # Set title
        if title:
            ax.set_title(title, fontname='Garamond', fontsize=fontsize)

        # Set legend
        if clusters and legend and list_legend is not None:
            ax.legend(handles=list_legend, loc='best', fontsize=int(fontsize * 0.8))

        # Get rid of the frame
        for place, spine in ax.spines.items():
            if place != 'bottom':
                spine.set_visible(False)
            else:
                spine.set_bounds(self.times[0], self.times[1])
                spine.set_color('#476b6b')

        ax.tick_params(top=False, bottom=True, right=False, left=True, labelleft=True, labelbottom=True)
        return ax

    def _plot_links(self, ax, lw_links=2.5, arrivals_marker=True):
        for k, lp in zip(self.links, self.link_presence):
            id1 = k[0]
            id2 = k[1]
            idmax = max(id1, id2)
            idmin = min(id1, id2)
            eps = random.choice([1, -1]) * 0.1
            ax.hlines([(idmax + idmin) / 2 + eps] * (len(lp) // 2), xmin=lp[::2], xmax=lp[1::2],
                      colors='k', linewidth=lw_links + 1, alpha=0.5)
            ax.vlines(lp[::2], ymin=idmin, ymax=idmax,
                      linewidth=lw_links, alpha=0.2)
            if arrivals_marker is True:
                ax.plot([lp[::2]], [idmin], color='#004d00', marker='^', alpha=1, markersize=7)
                ax.plot([lp[::2]], [idmax], color='#004d00', marker='v', alpha=1, markersize=7)

    def _plot_nodes(self, ax, nodes_cmap, lw_nodes=2.5, nodes_list=None):

        if nodes_list is not None:
            for n, np in zip(self.nodes, self.node_presence):
                if n in nodes_list:
                    coln = nodes_cmap[n]
                    ax.hlines([n] * (len(np) // 2), xmin=np[::2], linewidth=lw_nodes,
                              xmax=np[1::2], colors=coln, alpha=1)
        else:
            for n, np in zip(self.nodes, self.node_presence):
                coln = nodes_cmap[n]
                ax.hlines([n] * (len(np) // 2), xmin=np[::2], linewidth=lw_nodes,
                          xmax=np[1::2], colors=coln, alpha=1)

    def _plot_clusters(self, ax, clusters, cmap_clusters=None, lw_nodes=2.5,
                       legend=True, nodes_list=None):

        if type(clusters) in [dict, defaultdict]:

            if nodes_list is not None:
                for n, np in zip(self.nodes, self.node_presence):
                    if n in nodes_list:
                        color_n = '#666699'  # c_map(p)
                        ax.hlines([n] * (len(np) // 2), xmin=np[::2], linewidth=lw_nodes,
                                  xmax=np[1::2], colors=color_n, alpha=0.9)
            else:
                for n, np in zip(self.nodes, self.node_presence):
                    color_n = '#666699'  # c_map(p)
                    ax.hlines([n] * (len(np) // 2), xmin=np[::2], linewidth=lw_nodes,
                              xmax=np[1::2], colors=color_n, alpha=0.9)

            if cmap_clusters is None:
                cmap_clusters = 'plasma'

            norm_colors = plt.Normalize(vmin=min(clusters.keys()), vmax=max(clusters.keys()))
            c_map_cluster = cm.get_cmap(cmap_clusters, len(clusters.keys()))

            if type(next(iter(next(iter(clusters.values()))))[2]) is str:
                label_to_node = {v: k for k, v in self.node_to_label.items()}
                clusters = {k: [(t0, t1, label_to_node[n]) for t0, t1, n in clusters[k]] for k in clusters}

            list_legend = []
            for d in sorted(clusters, reverse=True):
                current_color = c_map_cluster(norm_colors(d))
                if legend:
                    list_legend.append(mpatch.Patch(color=current_color, label=str(round(d, 2)), ))

                if nodes_list is not None:
                    for t0, t1, n in clusters[d]:
                        if n in nodes_list:
                            if t0 == t1:
                                ax.plot(t0, n, color=current_color, marker='o', alpha=1, markersize=lw_nodes * 1.5)
                            else:
                                ax.hlines(n, xmin=t0, linewidth=lw_nodes * 1.3,
                                          xmax=t1, color=current_color, alpha=1)
                else:
                    for t0, t1, n in clusters[d]:
                        if t0 == t1:
                            ax.plot(t0, n, color=current_color, marker='o', alpha=1, markersize=lw_nodes * 1.5)
                        else:
                            ax.hlines(n, xmin=t0, linewidth=lw_nodes * 1.3,
                                      xmax=t1, color=current_color, alpha=1)

            if legend is True:
                # Dirty trick to scale legend
                while len(list_legend) > 10:
                    list_legend = list_legend[:len(list_legend) - 1:2] + [list_legend[-1]]
                return list_legend

        elif type(clusters) is list:
            if nodes_list is not None:
                for n, np in zip(self.nodes, self.node_presence):
                    if n in nodes_list:
                        color_n = '#666699'  # c_map(p)
                        ax.hlines([n] * (len(np) // 2), xmin=np[::2], linewidth=lw_nodes,
                                  xmax=np[1::2], colors=color_n, alpha=0.9)
            else:
                for n, np in zip(self.nodes, self.node_presence):
                    color_n = '#666699'  # c_map(p)
                    ax.hlines([n] * (len(np) // 2), xmin=np[::2], linewidth=lw_nodes,
                              xmax=np[1::2], colors=color_n, alpha=0.9)

            if cmap_clusters is None:
                cmap_clusters = 'nipy_spectral'

            cmap_clusters_tmp = get_cmap(len(clusters) + 2, cmap=cmap_clusters)
            cmap_clusters = lambda x: cmap_clusters_tmp(x + 2)  # Add an offset to get rid of the first colors
            #
            if type(clusters[0][0][2]) is str:
                label_to_node = {v: k for k, v in self.node_to_label.items()}
                clusters = [[(t0, t1, label_to_node[n]) for t0, t1, n in c] for c in clusters]
            if nodes_list is not None:
                for c in range(len(clusters)):
                    for t0, t1, n in clusters[c]:
                        if n in nodes_list:
                            if t0 == t1:
                                ax.plot(t0, n, color=cmap_clusters(c), marker='o', alpha=1, markersize=lw_nodes * 1.5)
                            else:
                                ax.hlines([n], xmin=t0, linewidth=lw_nodes * 1.3,
                                          xmax=t1, color=cmap_clusters(c), alpha=1)
            else:
                for c in range(len(clusters)):
                    for t0, t1, n in clusters[c]:
                        if t0 == t1:
                            ax.plot(t0, n, color=cmap_clusters(c), marker='o', alpha=1, markersize=lw_nodes * 1.5)
                        else:
                            ax.hlines([n], xmin=t0, linewidth=lw_nodes * 1.3,
                                      xmax=t1, color=cmap_clusters(c), alpha=1)

        else:
            raise TypeError("Type " + str(type(clusters)) + " is not supported for plotting clusters.")

    def plot_3d(self, fontsize=26):
        """
        Experimental visualisation of a ``StreamGraph`` object in a three dimensionnal space.

        :param fontsize:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lnodes = max(self.nodes)
        c_map = get_cmap(lnodes)
        T = self.times
        ax.set_xlim(0, lnodes)
        ax.set_ylim(T[0], T[1])
        z_max = max([max([j - i for i, j in zip(t[::2], t[1::2])]) for t in self.link_presence])
        ax.set_zlim(0, z_max)
        y = [T[0], T[1]]

        for p, t in zip(self.nodes, self.node_presence):
            clr = c_map(p)
            # dotted line all the time corresponding to the existence of the node,
            # not his presence
            if len(self.nodes) < 350:
                ax.plot(xs=[p, p], ys=y, zs=0, zdir='z', linestyle='--', linewidth=0.7,
                        color=clr, alpha=0.1)
            for i, j in zip(t[::2], t[1::2]):
                # Line corresponding to the node presence
                ax.plot(xs=[p, p], ys=[i, j], zs=0, zdir='z', linewidth=1.1,
                        color=clr, alpha=0.9)
        for k, t in zip(self.links, self.link_presence):
            id1 = self.nodes.index(k[0])
            id2 = self.nodes.index(k[1])
            idmax = max(id1, id2)
            idmin = min(id1, id2)
            for i, j in zip(t[::2], t[1::2]):

                # eps = random.choice([1, -1]) * (random.random() / 5)
                id_tmp = (idmax + idmin) / 2  # + eps
                # Line corresponding to the duration of the link
                ax.plot(xs=[id_tmp, id_tmp], ys=[i, i], zs=[0, j - i], zdir='z',
                        color='#152686', linewidth=7, alpha=0.25)
                if len(self.links) < 150:
                    # Line corresponding to the link between two nodes
                    ax.plot(xs=[idmin, idmax], ys=[i, i], zs=0, zdir='z',
                            linewidth=1.5, alpha=0.15, color='#BDBDBD')
                    # Lines corresponding of both ending of the link to each node
                    ax.plot(xs=[idmax], ys=[i], zs=0, zdir='z', color='k', marker='o', alpha=0.2)
                    ax.plot(xs=[idmin], ys=[i], zs=0, zdir='z', color='k', marker='o', alpha=0.2)
        ax.set_title('3d representation of a Stream Graph', fontname='Garamond', fontsize=fontsize)
        ax.set_xticklabels(self.nodes, fontname='Garamond', fontsize=fontsize)
        ax.set_xlabel('Nodes', fontname='Garamond', fontsize=fontsize)
        ax.set_ylabel('Time', fontname='Garamond', fontsize=fontsize)
        ax.set_zlabel('Link\'s duration ', fontname='Garamond', fontsize=fontsize)
        ax.view_init(elev=20., azim=-35)
        return ax

    def plot_aggregated_graph(self, fontsize=26, title=None, lw_links=5, node_size=900,
                              layout=None):
        """
        Display the aggregated graph of a ``StreamGraph``

        :param fontsize:
        :return: A ``matplotlib`` figure object
        """
        fig = plt.figure()
        ax = plt.axes()
        G_glob = self.aggregated_graph(to_networkx=True)
        if layout is None:
            layout = nx.circular_layout
        pos = layout(G_glob)
        c_map = get_cmap(len(self.nodes) + 2)
        dict_node_colors = {n: c_map(p) for p, n in enumerate(G_glob.nodes())}

        nx.draw_networkx_labels(G_glob, pos, font_size=fontsize, ax=ax, font_family='Garamond')
        node_c = [dict_node_colors[n] for n in G_glob.nodes()]
        nx.draw_networkx_nodes(G_glob, pos, node_size=node_size,
                               node_color=node_c, alpha=0.5, ax=ax)
        nx.draw_networkx_edges(G_glob, pos, edge_color='#2d5986',
                               alpha=0.3, width=lw_links, ax=ax)
        for place, spine in plt.gca().spines.items():
            spine.set_visible(False)
        if title is not None:
            plt.title(title, fontname='Garamond', fontsize=fontsize)
        plt.tick_params(top=False, bottom=False, right=False, left=False, labelbottom=False, labelleft=False)
        return fig

    def plot_instant_graph(self, time=0, node_size=1000, fontsize=26, title=None, layout=None):
        """
        Plot the induced static graph at time *t*

        :param time: an instant (a timestamp)
        :param node_size: graphic parameters for the size of nodes
        :param fontsize:
        :return: A ``matplotlib`` figure object
        """
        fig = plt.figure()
        ax = plt.axes()
        G_glob = self.aggregated_graph(to_networkx=True)
        if layout is None:
            layout = nx.circular_layout
        pos = layout(G_glob)
        c_map = get_cmap(len(self.nodes) + 2)
        dict_node_colors = {n: c_map(p) for p, n in enumerate(G_glob.nodes())}

        nx.draw_networkx_labels(G_glob, pos, font_size=fontsize, ax=ax, font_family='Garamond')
        nx.draw_networkx_nodes(G_glob, pos, node_size=node_size, node_color='w', alpha=0, ax=ax)

        G = nx.from_dict_of_lists(self.instant_graph(time))
        node_c = [dict_node_colors[n] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_size,
                               node_color=node_c, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='#2d5986',
                               alpha=0.5, width=10, ax=ax)
        ax.set_xlabel("Time : {0}".format(time), fontname='Garamond', fontsize=fontsize, color='#476b6b')
        for place, spine in plt.gca().spines.items():
            spine.set_visible(False)
        if title is not None:
            plt.title(title, fontname='Garamond', fontsize=fontsize)
        plt.tick_params(top=False, bottom=False, right=False, left=False, labelbottom=False, labelleft=False)
        return fig

    def animated_plot(self, repeat=False, cmap='nipy_spectral', fontsize=26, layout=None):
        """
        Display an animation of a ``stream graph`` from its induced graphs

        :param repeat:
        :param cmap:
        :param fontsize:
        :return: 
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.plot(ax=ax1)

        interactions_times = self.event_times()
        line, = ax1.plot([], [], color='#B6AAC0', lw=8, alpha=0.5)

        G_glob = self.aggregated_graph(to_networkx=True)
        if layout is None:
            layout = nx.circular_layout
        pos = layout(G_glob)

        c_map = get_cmap(len(self.nodes) + 2, cmap=cmap)
        dict_node_colors = {n: c_map(p) for p, n in enumerate(G_glob.nodes())}

        # Limits of the first plot
        ax1.set_xlim((self.times[0] * 0.95,
                      self.times[1] * 1.05))
        ax1.set_ylim((-0.3, len(self.nodes) + 0.3))

        # # Limits of the second plot (circular layout)
        # ax2.set_xlim((-1.2, 1.2))
        # ax2.set_ylim((-1.2, 1.2))

        def init():
            # First Plot
            line.set_data([], [])
            # Second Plot
            nx.draw_networkx_labels(G_glob, pos, font_size=fontsize, ax=ax2, font_family='Garamond',
                                    font_color='k', alpha=1)
            nodes = nx.draw_networkx_nodes(G_glob, pos, node_size=1000, node_color='w', alpha=0, ax=ax2)
            return line, nodes,

        def update(t):
            # First Plot
            y = numpy.linspace(0, max(self.nodes), 1000)
            x = [t]
            ax1.set_xlabel("Time: " + str(t), fontname='Garamond', fontsize=fontsize, color='#476b6b')
            line.set_data(x, y)
            # Second Plot
            ax2.clear()
            G = self.instant_graph(t, to_networkx=True)
            node_c = [dict_node_colors[n] for n in G.nodes()]
            nx.draw_networkx_labels(G_glob, pos, font_size=fontsize, ax=ax2, font_family='Garamond',
                                    font_color='k', alpha=1)  # '#666699'
            _ = nx.draw_networkx_nodes(G_glob, pos, node_size=1000, node_color='w', alpha=0, ax=ax2)
            nodes = nx.draw_networkx_nodes(G, pos, node_size=1000,
                                           node_color=node_c, alpha=0.75, ax=ax2)
            edges = nx.draw_networkx_edges(G, pos, edge_color='#2d5986',
                                           alpha=0.5, width=6, ax=ax2)
            ax2.set_xlabel("Time: " + str(t), fontname='Garamond', fontsize=fontsize, color='#476b6b')
            if not nodes:
                nodes = nx.draw_networkx_nodes(G_glob, pos, node_size=1000, node_color='w', alpha=0, ax=ax2)
            if not G.edges():
                return line, nodes,
            return line, nodes, edges,  # labels

        for place, spine in plt.gca().spines.items():
            spine.set_visible(False)

        plt.tick_params(top=False, bottom=False, right=False, left=False, labelbottom=False, labelleft=False)
        frms = list(sorted(interactions_times))
        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       frames=frms, interval=2000,
                                       blit=True, repeat=repeat)
        return anim

    def get_card_W(self):
        """
        Return the sum of temporal nodes presence time ($|W|$)

        :return:
        """
        return sum([t1 - t0 for i in self.node_presence
                    for t0, t1 in zip(i[::2], i[1::2])])

    def get_card_E(self):
        """
        Return the sum temporal links presence time ($|E|$)

        :return:
        """
        return sum([t1 - t0 for i in self.link_presence
                    for t0, t1 in zip(i[::2], i[1::2])])

    def duration(self):
        """
        Return the total duration of a ``stream graph`` object

        :return:
        """
        return self.times[1] - self.times[0]

    def surface(self):
        """
        Return the surface of a ``stream graph`` object

        :return:
        """
        return sum([t1 - t0 for v in self.link_presence for t0, t1 in zip(v[::2], v[1::2])])

    def coverage(self):
        """
        Return the coverage of a ``stream graph`` object

        :return:
        """
        T = self.duration()
        card_W = self.get_card_W()
        cov = card_W / (len(self.nodes) * T)
        return cov

    def nb_nodes(self):
        """
        The number nodes according to their presence in the stream graph

        :return:
        """
        T = self.times[1] - self.times[0]
        card_W = self.get_card_W()
        nb_nodes = card_W / T
        return nb_nodes

    def nb_links(self):
        """
        The number of links according to their presence in the stream graph

        :return:
        """
        T = self.times[1] - self.times[0]
        card_E = self.get_card_E()
        nb_links = card_E / T
        return nb_links

    def nodes_over_time(self, to_series=True, datetime=True, events=None, eps=None):
        """
        Compute the number of nodes over time in the Stream Graph.

        :param to_series: return a ``pandas`` series
        :param datetime: A boolean value to display timestamp as readable data
        :param events:
        :param eps:
        :return: A dictionnary linking event time to the corresponding number of nodes
        """
        t_to_nb_nodes = {}
        cnt_nodes = 0
        t_last_event = None
        if events is None:
            events = self.ordered_events()
        if eps is None:
            eps = min([e[2] - e[1] for e in events if (e[0] in {1, 2} and e[2] - e[1] != 0)]) * (10 ** -3)
        for e in events:
            c = e[0]
            t = e[1]
            if c == 2:
                _, t0, t1, n = e

                if t != t_last_event:
                    t_to_nb_nodes[t - eps] = cnt_nodes
                    t_last_event = t

                cnt_nodes += 1
                t_to_nb_nodes[t] = cnt_nodes

            elif c == -2:
                _, t1, n = e

                if t != t_last_event:
                    t_to_nb_nodes[t - eps] = cnt_nodes
                    t_last_event = t

                cnt_nodes -= 1
                t_to_nb_nodes[t + eps] = cnt_nodes
        if to_series:
            if datetime:
                Index = [dt.datetime.fromtimestamp(ts) for ts in t_to_nb_nodes.keys()]
            else:
                Index = list(t_to_nb_nodes.keys())
            t_to_nb_nodes = pd.Series(list(t_to_nb_nodes.values()), index=Index)

        return t_to_nb_nodes

    def links_over_time(self, to_series=True, datetime=True, events=None, eps=None):
        """
        Compute the number of active links over time in the Stream Graph.

        :param to_series: return a ``pandas`` series
        :param datetime: A boolean value to display timestamp as readable data
        :param events:
        :param eps:
        :return: A dictionnary event time to the corresponding number of nodes
        """
        t_to_nb_links = {}
        cnt_links = 0
        t_last_event = None
        if events is None:
            events = self.ordered_events()
        if eps is None:
            eps = min([e[2] - e[1] for e in events if (e[0] in {1, 2} and e[2] - e[1] != 0)]) * (10 ** -3)
        for e in events:
            c = e[0]
            t = e[1]
            if c == 1:
                _, t0, t1, u, v = e

                if t != t_last_event:
                    t_to_nb_links[t - eps] = cnt_links
                    t_last_event = t

                cnt_links += 1
                t_to_nb_links[t] = cnt_links

            elif c == -1:
                _, t1, u, v = e

                if t != t_last_event:
                    t_to_nb_links[t - eps] = cnt_links
                    t_last_event = t

                cnt_links -= 1
                t_to_nb_links[t + eps] = cnt_links

        if to_series:
            if datetime:
                Index = [dt.datetime.fromtimestamp(ts) for ts in t_to_nb_links.keys()]
            else:
                Index = list(t_to_nb_links.keys())
            t_to_nb_links = pd.Series(list(t_to_nb_links.values()), index=Index)

        return t_to_nb_links

    def degree_over_time(self, nodes=None, to_series=True, datetime=True, events=None, eps=None):
        """
        Compute the degree over time (of the considered ``nodes``) in the Stream Graph

        :param nodes:
        :param to_series:
        :param datetime:
        :param events:
        :param eps:
        :return: A dictionary event time to counter of nodes degrees
        """
        if nodes is None:
            nodes = set(self.nodes)
        if type(nodes) == list:
            nodes = set(nodes)
        t_to_node_degree = {}
        counter_degree = Counter()
        t_last_event = None
        if events is None:
            events = self.ordered_events()
        if eps is None:
            eps = min([e[2] - e[1] for e in events if (e[0] in {1, 2} and e[2] - e[1] != 0)]) * (10 ** -3)
        for e in events:
            c = e[0]
            t = e[1]
            if c == 1:
                _, _, _, u, v = e
                if u in nodes or v in nodes:
                    if t != t_last_event:
                        t_to_node_degree[t - eps] = counter_degree.copy()
                        t_last_event = t
                    if u in nodes:
                        counter_degree[u] += 1
                    if v in nodes:
                        counter_degree[v] += 1
                    t_to_node_degree[t] = counter_degree.copy()

            elif c == -1:
                _, _, u, v = e
                if u in nodes or v in nodes:
                    if t != t_last_event:
                        t_last_event = t
                        t_to_node_degree[t - eps] = counter_degree.copy()
                    if u in nodes:
                        counter_degree[u] -= 1
                    if v in nodes:
                        counter_degree[v] -= 1
                    t_to_node_degree[t + eps] = counter_degree.copy()

            elif c == 2:
                _, _, _, n = e
                if n in nodes:
                    if t != t_last_event:
                        t_last_event = t
                        t_to_node_degree[t - eps] = counter_degree.copy()
                    counter_degree[n] = 0
                    t_to_node_degree[t] = counter_degree.copy()

            elif c == -2:
                _, _, n = e
                if n in nodes:
                    if t != t_last_event:
                        t_last_event = t
                        t_to_node_degree[t - eps] = counter_degree.copy()
                    del counter_degree[n]
                    t_to_node_degree[t + eps] = counter_degree.copy()

        if to_series:
            if datetime:
                Index = [dt.datetime.fromtimestamp(ts) for ts in t_to_node_degree.keys()]
            else:
                Index = list(t_to_node_degree.keys())
            t_to_node_degree = {v: pd.Series([t_to_node_degree[t][v]
                                              if v in t_to_node_degree[t]
                                              else numpy.nan for t in t_to_node_degree],
                                             index=Index) for v in nodes}

        return t_to_node_degree

    def mean_degree_over_time(self, to_series=True, datetime=True, events=None, eps=None):
        """
        Compute the mean degree over time in the Stream Graph.

        :param to_series:
        :param datetime:
        :param events:
        :param eps:
        :return: A dictionnary event time to the corresponding number of nodes
        """
        t_to_mean_degree = {}
        cnt_nodes = 0
        cnt_global_degree = 0
        t_last_event = None
        if events is None:
            events = self.ordered_events()
        if eps is None:
            eps = min([e[2] - e[1] for e in events if (e[0] in {1, 2} and e[2] - e[1] != 0)]) * (10 ** -3)
        for e in events:
            c = e[0]
            t = e[1]
            if c == 1:
                _, t0, t1, u, v = e

                if t != t_last_event:
                    t_to_mean_degree[t - eps] = cnt_global_degree / cnt_nodes
                    t_last_event = t

                cnt_global_degree += 2
                t_to_mean_degree[t] = cnt_global_degree / cnt_nodes
            elif c == -1:
                _, t1, u, v = e

                if t != t_last_event:
                    t_to_mean_degree[t - eps] = cnt_global_degree / cnt_nodes
                    t_last_event = t

                cnt_global_degree -= 2
                t_to_mean_degree[t + eps] = cnt_global_degree / cnt_nodes

            elif c == 2:
                _, t0, t1, n = e

                if t != t_last_event:
                    if cnt_nodes > 0:
                        t_to_mean_degree[t - eps] = cnt_global_degree / cnt_nodes
                    else:
                        t_to_mean_degree[t - eps] = 0
                    t_last_event = t

                cnt_nodes += 1
                if cnt_nodes > 0:
                    t_to_mean_degree[t] = cnt_global_degree / cnt_nodes
                else:
                    t_to_mean_degree[t] = 0

            elif c == -2:
                _, t1, n = e

                if t != t_last_event:
                    t_to_mean_degree[t - eps] = cnt_global_degree / cnt_nodes
                    t_last_event = t

                cnt_nodes -= 1
                if cnt_nodes > 0:
                    t_to_mean_degree[t + eps] = cnt_global_degree / cnt_nodes
                else:
                    t_to_mean_degree[t + eps] = 0

        if to_series:
            if datetime:
                Index = [dt.datetime.fromtimestamp(ts) for ts in t_to_mean_degree.keys()]
            else:
                Index = list(t_to_mean_degree.keys())
            t_to_mean_degree = pd.Series(list(t_to_mean_degree.values()), index=Index)

        return t_to_mean_degree

    def node_weight_at_t(self, t):
        """
        Return the weight of nodes at instant ``t``

        :param t:
        :return:
        """
        V = len(self.nodes)
        Vt = sum([1 for i in self.node_presence for ta, tb in zip(i[::2], i[1::2]) if ta <= t <= tb])
        return Vt / V

    def link_weight_at_t(self, t):
        """
        Return the weight of links at instant ``t``

        :param t:
        :return:
        """
        l = len(self.nodes)
        E = (l * (l - 1)) / 2
        Et = sum([1 for i in self.link_presence for ta, tb in zip(i[::2], i[1::2]) if ta <= t <= tb])
        return Et / E

    def plot_link_weight(self):
        """
        Display the link weight over time

        :return:
        """
        fig = plt.figure()
        T = self.event_times()
        # add epsilon to capture information just before an event and just after
        epsilon = 10 ** -9
        resampled_T = []
        for t in T:
            resampled_T.append(t - epsilon)
            resampled_T.append(t)
            resampled_T.append(t + epsilon)
        resampled_T.sort()
        lt = [self.link_weight_at_t(t) for t in resampled_T]
        plt.plot(resampled_T, lt)
        plt.xlabel("Time")
        plt.ylabel("Link weight")
        plt.title("Link weight through time")
        return fig

    def node_weight_on_I(self, I):
        """
        Display the node weigth over a given interval ``I``

        :param I:
        :return:
        """
        V = len(self.nodes)
        Vt = [sum([1 for i in self.node_presence for ta, tb in zip(i[::2], i[1::2]) if ta <= t <= tb]) / V
              for t in I]
        return Vt

    def link_weight_on_I(self, I):
        """
        Display the link weigth over a given interval ``I``

        :param I:
        :return:
        """
        l = len(self.nodes)
        E = (l * (l - 1)) / 2
        Et = [sum([1 for i in self.link_presence for ta, tb in zip(i[::2], i[1::2]) if ta <= t <= tb]) / E
              for t in I]
        return Et

    def plot_node_weight(self):
        """
        Display the node weight over time

        :return:
        """
        fig = plt.figure()
        T = self.event_times()
        # add delta to capture information just before an event and just after
        epsilon = 10 ** -9
        resampled_T = []
        for t in T:
            resampled_T.append(t - epsilon)
            resampled_T.append(t)
            resampled_T.append(t + epsilon)
        resampled_T.sort()
        kt = [self.node_weight_at_t(t) for t in resampled_T]
        plt.plot(resampled_T, kt)
        plt.xlabel("Time")
        plt.ylabel("Node weight")
        plt.title(" Node weight through time")

        return fig

    def node_duration(self):
        """
        The node duration of a ``StreamGraph``

        :return: a float
        """
        card_W = self.get_card_W()
        nd = card_W / len(self.nodes)
        return nd

    def link_duration(self):
        """
        The link duration of a ``StreamGraph``

        :return: a float
        """
        l = len(self.nodes)
        possible_pairs = (l * (l - 1)) / 2
        card_E = self.get_card_E()
        return card_E / possible_pairs

    def _get_sum_intersection(self):
        l = len(self.nodes)
        sum_node_intersection = sum([min(a1, b1) - max(a0, b0)
                                     for i1 in range(l)
                                     for i2 in range(i1 + 1, l)
                                     for a0, a1 in zip(self.node_presence[i1][::2], self.node_presence[i1][1::2])
                                     for b0, b1 in zip(self.node_presence[i2][::2], self.node_presence[i2][1::2])
                                     if (a0 <= b1 and b0 <= a1)
                                     ])
        return sum_node_intersection

    def _get_sum_union(self):
        l = len(self.nodes)
        sum_node_union = sum([max(a1, b1) - min(a0, b0)
                              if (a0 <= b1 and b0 <= a1)
                              else (b1 - b0) + (a1 - a0)
                              for i1 in range(l)
                              for i2 in range(i1 + 1, l)
                              for a0, a1 in zip(self.node_presence[i1][::2], self.node_presence[i1][1::2])
                              for b0, b1 in zip(self.node_presence[i2][::2], self.node_presence[i2][1::2])
                              ])
        return sum_node_union

    def uniformity(self):
        """
        Return the uniformity of a ``StreamGraph`` object

        :return: the uniformity as a float
        """
        sum_node_intersection = self._get_sum_intersection()
        sum_node_union = self._get_sum_union()
        return sum_node_intersection / sum_node_union

    def density(self):
        """
        The density of the `StreamGraph`` object (aka the probability
        if we randomly chose two nodes that they are linked together)

        :return: the density as a float
        """
        l = len(self.nodes)
        card_E = self.get_card_E()
        sum_node_intersection = self._get_sum_intersection()
        return card_E / sum_node_intersection

    def link_densities(self):
        """
        Return the link densities of a ``StreamGraph`` object

        :return: The link's densities, each link is associated to a float
        """
        densities = Counter()
        for l, lp in zip(self.links, self.link_presence):
            n1 = l[0]
            n2 = l[1]
            Tuv = sum([t1 - t0 for t0, t1 in zip(lp[::2], lp[1::2])])
            # Pour que ça marche il faut que les noeuds soient représentés par leurs indices dans la liste
            intersection_Tu_Tv = sum([min(a1, b1) - max(a0, b0)
                                      for a0, a1 in zip(self.node_presence[n1][::2], self.node_presence[n1][1::2])
                                      for b0, b1 in zip(self.node_presence[n2][::2], self.node_presence[n2][1::2])
                                      if (a0 <= b1 and b0 <= a1)
                                      ])
            densities[l] = Tuv / intersection_Tu_Tv
        return densities

    def node_densities(self):
        """
        Return the node densities of a ``StreamGraph`` object

        :return:
        """
        densities = Counter()
        for n, np in zip(self.nodes, self.node_presence):
            sum_links = sum([t1 - t0
                             for l, lp in zip(self.links, self.link_presence)
                             if (l[0] == n or l[1] == n)
                             for t0, t1 in zip(lp[::2], lp[1::2])
                             ])

            sum_node_intersection = sum([min(a1, b1) - max(a0, b0)
                                         for n2, np2 in zip(self.nodes, self.node_presence)
                                         if (n2 != n)
                                         for a0, a1 in zip(np[::2], np[1::2])
                                         for b0, b1 in zip(np2[::2], np2[1::2])
                                         if (a0 <= b1 and b0 <= a1)
                                         ])
            densities[n] = sum_links / sum_node_intersection
        return densities

    def neighborhood(self):
        """
        Neighborhood of each node (each node --> their neighbors with the associated interval of time
        (corresponding to sum of temporal link))

        :return: the neighborhood of each node in a dictionary
        """
        N = {n: {} for n in self.nodes}
        for l, lp in zip(self.links, self.link_presence):
            n1 = l[0]
            n2 = l[1]
            N[n2][n1] = lp
            N[n1][n2] = lp
        return N

    def neighborhood_from_node(self, node):
        """
        The neighborhood of a specific node (node <-> their neighbor with the associated interval of time)

        :param node: a node of the ``StreamGraph``
        :return: The neighborhood of a specific node in a dictionary
        """
        N = {}
        for l, lp in zip(self.links, self.link_presence):
            n1 = l[0]
            n2 = l[1]
            if n1 == node:
                N[n2] = lp
            if n2 == node:
                N[n1] = lp
        return N

    def degrees(self):
        """
        Each node degree in the ``StreamGraph``

        :return: A dictionary linking each node to its degree
        """
        T = self.times[1] - self.times[0]
        degrees = Counter()
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            s_d = sum([t1 - t0 for t0, t1 in zip(lp[::2], lp[1::2])]) / T
            degrees[u] += s_d
            degrees[v] += s_d
        return degrees

    def nb_neighbors(self):
        """
        Return the number of nieghbors of each node

        :return:
        """
        nb_neighbors = Counter()
        for l in self.links:
            nb_neighbors[l[0]] += 1
            nb_neighbors[l[1]] += 1
        return nb_neighbors

    def sum_degree_neighborhood(self, degrees=None):
        if not degrees:
            degrees = self.degrees()
        sum_degree_neighborhood = Counter()
        for l, lp in zip(self.links, self.link_presence):
            n1 = l[0]
            n2 = l[1]
            sum_degree_neighborhood[n1] += degrees[n2]
            sum_degree_neighborhood[n2] += degrees[n1]
        return sum_degree_neighborhood

    def mean_degree_neighborhood(self, degrees=None):
        S = self.sum_degree_neighborhood(degrees)
        N = self.nb_neighbors()
        mean_degree_neighborhood = Counter()
        # std_degree_neighborhood = Counter()
        for n in self.nodes:
            mean_degree_neighborhood[n] = S[n] / N[n]
        return mean_degree_neighborhood  # ,std_degree_neighborhood

    def expected_node_degrees(self, d=None):
        T = self.times[1] - self.times[0]
        expected_degrees = Counter()
        if not d:
            d = self.degrees()
        for n, np in zip(self.nodes, self.node_presence):
            expected_degrees[n] = (d[n] * T) / sum([t1 - t0 for t0, t1 in zip(np[::2], np[1::2])])
        return expected_degrees

    def average_degree(self, degrees=None, nb_nodes=None):
        """
        The average degree of a stream graph

        :param degrees:
        :param nb_nodes:
        :return:
        """
        if not degrees:
            degrees = self.degrees()
        if not nb_nodes:
            nb_nodes = self.nb_nodes()
        T = self.times[1] - self.times[0]
        d_bar = sum([degrees[n] * sum([tb - ta for ta, tb in zip(np[::2], np[1::2])])
                     for n, np in enumerate(self.node_presence)
                     ]) / (nb_nodes * T)
        return d_bar

    def stream_graph_degree(self):
        T = self.times[1] - self.times[0]
        card_E = self.get_card_E()
        return card_E / (len(self.nodes) * T)

    def expected_stream_graph_degree(self):
        card_W = self.get_card_W()
        card_E = self.get_card_E()
        return 2 * card_E / card_W

    def clustering_coefficient(self):
        """
        A dictionary with for each node his clustering coefficient

        :return:
        """
        cc = Counter()
        N = self.neighborhood()
        for current_node in N:
            current_neighbors = N[current_node]
            l = len(current_neighbors)
            list_neighbors = list(current_neighbors)
            sum_link_intersection = sum([min(a1, b1) - max(a0, b0)
                                         for i1 in range(l)
                                         for i2 in range(i1 + 1, l)
                                         for a0, a1 in zip(current_neighbors[list_neighbors[i1]][::2],
                                                           current_neighbors[list_neighbors[i1]][1::2])
                                         for b0, b1 in zip(current_neighbors[list_neighbors[i2]][::2],
                                                           current_neighbors[list_neighbors[i2]][1::2])
                                         if (b1 >= a0 and a1 >= b0)
                                         ])
            sum_triplet_intersection = sum([
                min(a1, b1, c1) - max(a0, b0, c0)
                for i1 in range(l)
                for i2 in range(i1 + 1, l)
                for a0, a1 in zip(current_neighbors[list_neighbors[i1]][::2],
                                  current_neighbors[list_neighbors[i1]][1::2])
                for b0, b1 in zip(current_neighbors[list_neighbors[i2]][::2],
                                  current_neighbors[list_neighbors[i2]][1::2])
                if list_neighbors[i2] in N[list_neighbors[i1]]
                for c0, c1 in zip(N[list_neighbors[i1]][list_neighbors[i2]][::2],
                                  N[list_neighbors[i1]][list_neighbors[i2]][1::2])
                if (a1 >= b0 and a1 >= c0 and b1 >= a0 and b1 >= c0 and c1 >= b0 and c1 >= a0)
            ])
            # Last index on neighbors can be inverted if u is a neighbor of v,
            #  v is a neighbor of u
            if sum_link_intersection != 0:
                cc[current_node] = sum_triplet_intersection / sum_link_intersection
            else:
                cc[current_node] = 0
        return cc

    def average_clustering(self, cc=None):
        """
        The average clustering coefficient of a stream graph

        :return:
        """
        if not cc:
            cc = self.clustering_coefficient()
        T = self.times[1] - self.times[0]
        cc_bar = sum([coef * sum([tb - ta for ta, tb in zip(i[::2], i[1::2])])
                      for i, coef in zip(self.node_presence, cc.values())]) / (self.nb_nodes() * T)
        return cc_bar

    def induced_line_stream(self):
        """
        The induced line stream (which is a stream graph too) corresponding to the stream graph

        :return:
        """
        node_to_label = {i: l for i, l in enumerate(self.links)}
        label_to_node = {v: k for k, v in node_to_label.items()}
        induced_nodes = self.links
        induced_links = []
        link_presence = []
        for n1 in induced_nodes:
            for n2 in induced_nodes:
                if n1 == n2:
                    continue
                if n1[0] in n2 or n1[1] in n2:
                    if (n1, n2) in induced_links or (n2, n1) in induced_links:
                        continue
                    id1 = self.links.index(n1)
                    id2 = self.links.index(n2)
                    intersection_time = []
                    for t1a, t1b in zip(self.link_presence[id1][::2], self.link_presence[id1][1::2]):
                        for t2a, t2b in zip(self.link_presence[id2][::2], self.link_presence[id2][1::2]):
                            if (min(t1b, t2b) - max(t1a, t2a)) > 0:
                                intersection_time += [max(t1a, t2a)]
                                intersection_time += [min(t1b, t2b)]
                    if intersection_time:
                        # print("n1 :", n1, "n2 :", n2)
                        induced_links.append((label_to_node[n1], label_to_node[n2]))
                        link_presence.append(intersection_time)
                        # print("induced links :", induced_links)
        LS = StreamGraph(times=self.times,
                         nodes=list(node_to_label.keys()),
                         node_presence=self.link_presence,
                         links=induced_links,
                         link_presence=link_presence
                         )
        return LS

    def number_of_link_per_node(self, dataframe=False):
        d = {n: 0 for n in self.nodes}
        for l, lp in zip(self.links, self.link_presence):
            s = len(lp) / 2
            d[l[0]] += s
            d[l[1]] += s
        if dataframe:
            D = pd.DataFrame.from_dict(d, orient='index', dtype=int)

            return D
        else:
            return d

    def number_of_pair_per_link(self, dataframe=False):
        d = {n: 0 for n in self.links}
        for l, lp in zip(self.links, self.link_presence):
            s = len(lp) / 2
            d[l] += s
        if dataframe:
            D = pd.DataFrame.from_dict(d, orient='index', dtype=int)

            return D
        else:
            return d

    def transform_links_label_to_int(self, index=False):
        dict_node_label_2_int = {}
        dict_int_2_node_label = {}
        for i, n in enumerate(self.nodes):
            dict_node_label_2_int[n] = i
            dict_int_2_node_label[i] = n
            self.nodes[i] = i
        for l, j in zip(self.links, range(len(self.links))):
            self.links[j] = (dict_node_label_2_int[l[0]], dict_node_label_2_int[l[1]])

        self.node_to_label = dict_int_2_node_label
        print("NODE LABEL :", self.node_to_label)

        if index:
            return dict_int_2_node_label, dict_node_label_2_int

    #####################################################
    #           Methods to suppress nodes or links      #
    #####################################################

    def remove_node(self, n):
        """
        Remove the node *n* from the Stream Graph

        :param n: a node
        :return: 
        """
        self.remove_nodes({n})

    def remove_nodes(self, nodes_to_remove):
        """
        Remove all nodes in *nodes_to_remove* from the Stream Graph

        :param nodes_to_remove: a set of nodes
        :return:
        """
        if type(next(iter(nodes_to_remove))) is not int:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            nodes_to_remove = [label_to_node[lab] for lab in nodes_to_remove]

        nodes_to_remove = set(nodes_to_remove)
        # Adjust node offset due to deletion
        node_to_new_id = {}
        offset = 0
        for n in self.nodes:
            if n in nodes_to_remove:
                offset += 1
            else:
                node_to_new_id[n] = n - offset

        # Adjust nodes label
        if self.node_to_label:
            new_labels = self.node_to_label.copy()
            for u in self.nodes:
                if u not in nodes_to_remove:
                    new_labels[node_to_new_id[u]] = self.node_to_label[u]
            self.node_to_label = new_labels

        # Remove every links where an extremity is a node to remove
        id_to_remove = set()
        for id_l, l in enumerate(self.links):
            u, v = l
            if u in nodes_to_remove or v in nodes_to_remove:
                id_to_remove.add(id_l)
            else:
                new_u, new_v = node_to_new_id[u], node_to_new_id[v]
                self.links[id_l] = new_u, new_v

        self.links = [i for j, i in enumerate(self.links) if j not in id_to_remove]
        self.link_presence = [i for j, i in enumerate(self.link_presence) if j not in id_to_remove]

        self.nodes = self.nodes[:-len(nodes_to_remove)]
        self.node_presence = [el for j, el in enumerate(self.node_presence) if j not in nodes_to_remove]

    def remove_link(self, l):
        """
        Remove the (link,time_presence) from the Stream Graph

        :param l: 
        :return: 
        """
        if type(l[0]) is not int:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            l = (label_to_node[l[0]], label_to_node[l[1]])

        del self.link_presence[self.links.index(l)]
        self.links.remove(l)

    def remove_links(self, links_to_remove):
        """
        Remove links in the set *links_to_remove* from the Stream Graph

        :param links_to_remove:
        :return:
        """
        links_to_remove = set(links_to_remove)
        id_to_remove = set()
        for id_l, l in enumerate(self.links):
            if l in links_to_remove:
                id_to_remove.add(id_l)

        self.links = [i for j, i in enumerate(self.links) if j not in id_to_remove]
        self.link_presence = [i for j, i in enumerate(self.link_presence) if j not in id_to_remove]

    #########################################
    #       Connected Components            #
    #########################################

    def segmented_neighborhood(self):
        """
        On veut des segments de noeuds comme clefs ainsi que dans les voisins (b,e,v) : [(t0,t1,(nt0,nt1,u))] :)

        :return:
        """
        # See 'neighborhood_with_node_presence' in stream.py
        N = defaultdict(SortedSet)
        for l, lp in zip(self.links, self.link_presence):
            for t0, t1 in zip(lp[::2], lp[1::2]):
                nu, nv = self.get_according_node_presence_from_link(l, t0, t1)
                if nu and nv:
                    N[nu].add((t0, t1, nv))
                    N[nv].add((t0, t1, nu))
        return N

    def get_node_presence_from_interval(self, n, b, e):
        """
        Return the maximal segmented node corresponding to (b,e,n)

        :param n: node
        :param b: beginning of the interval (time)
        :param e: ending of the interval (time)
        :return: Maximal segmented node presence corresponding to (b,e,n) : (t0,t1)
        """
        for t0, t1 in zip(self.node_presence[n][::2], self.node_presence[n][1::2]):
            if b <= t0 and t1 <= e:
                return t0, t1
        return None

    def get_according_node_presence_from_link(self, l, t0, t1):
        """
        Get both extremities (which are maximal segmented nodes) of the link.

        :param l:
        :param t0: Beginning of the link
        :param t1: Ending of the link
        :return:
        """
        u, v = l[0], l[1]
        nu, nv = None, None
        for a, b in zip(self.node_presence[u][::2], self.node_presence[u][1::2]):
            if a <= t0 and t1 <= b:
                nu = (a, b, u)
                break
        for a, b in zip(self.node_presence[v][::2], self.node_presence[v][1::2]):
            if a <= t0 and t1 <= b:
                nv = (a, b, v)
                break
        return nu, nv

    def neighborhood_with_node_presence(self):
        """
        The neighborhood of each node globally

        :return: a dictionary (each node --> their neighbors with their presences)
        """
        N_presence = {}
        node_to_segmented_node = {}
        for e in self.ordered_events():
            c = e[0]
            if c == 2:
                _, t0, t1, n = e
                node_to_segmented_node[n] = (t0, t1, n)
                N_presence[(t0, t1, n)] = []
            elif c == 1:
                _, t0, t1, u, v = e
                u, v = node_to_segmented_node[u], node_to_segmented_node[v]
                N_presence[u].append(v)
                N_presence[v].append(u)
        return N_presence

    ####################################################################
    #                       Sort Functions                             #
    ####################################################################

    def node_position_by_arrival(self):
        """
        Sort nodes by their first time of arrival

        :return:
        """
        node_to_pos = {}
        time_of_arrival = {}
        for n, np in zip(self.nodes, self.node_presence):
            time_of_arrival[n] = min(np)
        cnt_pos = 0
        for n in time_of_arrival:
            node_to_pos[self.node_to_id[n]] = cnt_pos
            cnt_pos += 1
        return node_to_pos

    def node_position_by_increasing_degree(self, degrees_partition=None):
        """
        Sort nodes by their degree ;

        :param degrees_partition:
        :return:
        """
        if degrees_partition is None:
            degrees_partition = self.degrees_partition()
            isolated_nodes = self.isolated_nodes()
            degrees_partition[0] = set(tuple(isolated_nodes))
        node_to_pos = {}
        cnt_pos = 0
        for d in sorted(degrees_partition):
            sum_interval_degrees = Counter()
            for e in degrees_partition[d]:
                (t0, t1, n) = e
                sum_interval_degrees[n] += t1 - t0
            for n in sorted(sum_interval_degrees, key=sum_interval_degrees.get):
                node_to_pos[self.node_to_id[n]] = cnt_pos
                cnt_pos += 1
        return node_to_pos

    def node_position_by_peak_degree_arrival(self, degrees_partition=None):
        """
        Sort nodes increasingly by their maximal value of degree arrival time

        :return:
        """
        if degrees_partition is None:
            degrees_partition = self.degrees_partition()
            isolated_nodes = self.isolated_nodes()
            degrees_partition[0] = set(tuple(isolated_nodes))
        node_to_pos = {}
        cnt_pos = len(self.nodes) - 1
        visited = set()
        peak_degree_arrival = Counter()
        for d in sorted(degrees_partition, reverse=True):
            for e in degrees_partition[d]:
                (t0, t1, n) = e
                if n not in visited:
                    if n in peak_degree_arrival:
                        peak_degree_arrival[n] = min(peak_degree_arrival[n], t0)
                    else:
                        peak_degree_arrival[n] = t0
            for n in peak_degree_arrival:
                visited.add(n)
        for n in sorted(peak_degree_arrival, key=peak_degree_arrival.get):
            node_to_pos[self.node_to_id[n]] = cnt_pos
            cnt_pos -= 1
        return node_to_pos

    def link_position_by_duration(self):
        """
        sort links by their duration

        :return:
        """
        link_to_pos = {}
        link_duration = Counter()
        for l, lp in zip(self.links, self.link_presence):
            link_duration[l] = sum([t0 - t1 for t0, t1 in zip(lp[::2], lp[1::2])])
        cnt_pos = 0
        for l in sorted(link_duration, key=link_duration.get):
            link_to_pos[(self.node_to_id[l[0]], self.node_to_id[l[1]])] = cnt_pos
            cnt_pos += 1
        return link_to_pos

    def link_position_by_arrival(self):
        """
        sort links by their arrival

        :return:
        """
        link_to_pos = {}
        time_of_arrival = {}
        for l, lp in zip(self.links, self.link_presence):
            time_of_arrival[l] = min(lp)
        cnt_pos = 0
        for l in time_of_arrival:
            link_to_pos[(self.node_to_id[l[0]], self.node_to_id[l[1]])] = cnt_pos
            cnt_pos += 1
        return link_to_pos

    ####################################################################
    #       Induced Graphs and Substreams                              #
    ####################################################################

    def aggregated_graph(self, label=True, to_networkx=False, to_networkit=False):
        """
        Return the aggregated induced graph.

        :param to_networkit:
        :param to_networkx:
        :param label: True if we want the node's label in the adjacency list.
        :return: An adjacency list (networkx compatible)
        """
        if label:
            adjacency_list = {self.node_to_label[n]: [] for n in self.nodes}
            for l in self.links:
                n1 = self.node_to_label[l[0]]
                n2 = self.node_to_label[l[1]]
                adjacency_list[n1].append(n2)
                adjacency_list[n2].append(n1)
        else:
            adjacency_list = {n: [] for n in self.nodes}
            for l in self.links:
                n1 = l[0]
                n2 = l[1]
                adjacency_list[n1].append(n2)
                adjacency_list[n2].append(n1)
        if to_networkx:
            return nx.from_dict_of_lists(adjacency_list)
        if to_networkit:
            try:
                import networkit as nk
                G = nx.from_dict_of_lists(adjacency_list)
                return nk.nxadapter.nx2nk(G)
            except ImportError as e:
                raise ImportError("Package Networkit needed for this functionality")

        return adjacency_list

    def instant_graph(self, t, label=True, to_networkx=False, to_networkit=False):
        """
        Return an adjacency list corresponding to the induced graph
        from the stream graph at a specified time *t*.

        :param to_networkit:
        :param to_networkx:
        :param t: Time instant
        :param label: True if we want the node's label in the adjacency list.
        :return: An adjacency list (networkx compatible)
        """
        adjacency_list = defaultdict(list)
        for i, l in enumerate(self.links):
            u = l[0]
            v = l[1]
            for t0, t1 in zip(self.link_presence[i][::2], self.link_presence[i][1::2]):
                if t0 <= t <= t1:
                    if label:
                        adjacency_list[self.node_to_label[u]].append(self.node_to_label[v])
                        adjacency_list[self.node_to_label[v]].append(self.node_to_label[u])
                    else:
                        adjacency_list[u].append(v)
                        adjacency_list[v].append(u)
        # Add isolated nodes:
        for n, np in zip(self.nodes, self.node_presence):
            for t0, t1 in zip(np[::2], np[1::2]):
                if t0 <= t <= t1:
                    if label:
                        _ = adjacency_list[self.node_to_label[n]]  # Defaultdict functionality
                    else:
                        _ = adjacency_list[n]  # Defaultdict functionality
        if to_networkx:
            return nx.from_dict_of_lists(adjacency_list)
        if to_networkit:
            try:
                import networkit as nk
                G = nx.from_dict_of_lists(adjacency_list)
                return nk.nxadapter.nx2nk(G)
            except ImportError as e:
                raise ImportError("Package Networkit needed for this functionality")
        return adjacency_list

    def induced_substream_by_nodes(self, nodes_list):
        """
        Return the sub stream induced by *nodes_list*.
        IMPORTANT: Only include links with both extremities in *nodes_list*. (see filter_links otherwise)

        :param nodes_list: A list of nodes (Ids or Labels)
        :return: A Stream Graph
        """

        # Check if element of nodes_list are labels or nodes id:
        if type(nodes_list[0]) == str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            nodes_list = [label_to_node[n] for n in nodes_list]

        for n in nodes_list:
            if n not in self.nodes:
                raise ValueError("Trying to filter Stream Graph by nodes that does not exist in Stream Graph")

        new_nodes = []
        new_node_presence = [[] for _ in nodes_list]
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        new_links = []
        new_link_presence = []

        for n, np in zip(self.nodes, self.node_presence):
            if n in nodes_list:
                new_n = nodes_to_new_nodes[n]
                if self.node_to_label:
                    new_nodes_to_label[new_n] = self.node_to_label[n]
                if self.node_to_id:
                    new_nodes_to_id[new_n] = self.node_to_id[n]
                new_nodes.append(new_n)
                new_node_presence[new_n] = np
        new_nodes.sort()  # To corresponds to emplacement in new_node_presence

        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if u in nodes_list and v in nodes_list:
                new_u = nodes_to_new_nodes[u]
                new_v = nodes_to_new_nodes[v]
                new_links.append((new_u, new_v))
                new_link_presence.append(lp)

        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def induced_substream_by_links(self, links_list):
        """
        Return the sub stream induced by links in *links_list* and nodes implicated in these links.

        :param links_list: list of links ((id1,id2) or (label1,label2))
        :return: A Stream Graph
        """

        # Check if element of links_list are labels or nodes id:
        if type(links_list[0][0]) == str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            links_list = [(label_to_node[l[0]], label_to_node[l[1]]) for l in links_list]

        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        new_links = []
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if (u, v) in links_list or (v, u) in links_list:
                if u not in nodes_to_new_nodes:
                    new_u = nodes_to_new_nodes[u]
                    new_nodes_to_label[new_u] = self.node_to_label[u]
                    if self.node_to_id:
                        new_nodes_to_id[new_u] = self.node_to_id[u]
                    new_nodes.append(new_u)
                    new_node_presence.append(self.node_presence[u])
                else:
                    new_u = nodes_to_new_nodes[u]
                if v not in nodes_to_new_nodes:
                    new_v = nodes_to_new_nodes[v]
                    new_nodes_to_label[new_v] = self.node_to_label[v]
                    if self.node_to_id:
                        new_nodes_to_id[new_v] = self.node_to_id[v]
                    new_nodes.append(new_v)
                    new_node_presence.append(self.node_presence[v])
                else:
                    new_v = nodes_to_new_nodes[v]
                new_links.append((new_u, new_v))
                new_link_presence.append(lp)

        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def induced_substream_by_time_window(self, time_window):
        """
        Return the sub stream induced by T = [a,b].

        :param time_window: The time_window that delimit the substream
        :return: A Stream Graph
        """
        a, b = time_window
        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))

        new_links = []
        new_link_presence = []

        for n, np in zip(self.nodes, self.node_presence):
            new_np = []
            for t0, t1 in zip(np[::2], np[1::2]):
                if t0 <= b and a <= t1:  # Intersection
                    new_np += [max(a, t0), min(b, t1)]
            if new_np:
                new_n = nodes_to_new_nodes[n]
                if self.node_to_label:
                    new_nodes_to_label[new_n] = self.node_to_label[n]
                if self.node_to_id:
                    new_nodes_to_id[new_n] = self.node_to_id[n]
                new_nodes.append(new_n)
                new_node_presence.append(new_np)

        for l, lp in zip(self.links, self.link_presence):
            new_lp = []
            for t0, t1 in zip(lp[::2], lp[1::2]):
                if t0 <= b and a <= t1:
                    new_lp += [max(a, t0), min(b, t1)]
            if new_lp:
                u, v = l
                new_u = nodes_to_new_nodes[u]
                new_v = nodes_to_new_nodes[v]
                new_links.append((new_u, new_v))
                new_link_presence.append(new_lp)

        return StreamGraph(times=time_window,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def substream(self, cluster, substream_id=None, return_new_node_label=False):
        """
        Return the substream corresponding to the cluster:
        A Stream Graph containing the nodes of the cluster along with the segmented links that exist between them.

        :param return_new_node_label:
        :param cluster: A sequence of node segments: [(t0,t1,u),...,(t0,t1,v)]
        :param substream_id: (optional parameter) assign an id to the substream (usefull when dealing with components)
        :return:
        """

        if type(cluster[0][2]) == str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            cluster = [(t0, t1, label_to_node[n]) for t0, t1, n in cluster]

        new_nodes = []
        new_node_presence = []
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        new_nodes_to_label = {}
        new_nodes_to_id = {}

        for e in cluster:
            t0, t1, n = e
            if n in nodes_to_new_nodes:
                new_n = nodes_to_new_nodes[n]

                new_node_presence[new_n] += [t0, t1]
            else:
                new_n = nodes_to_new_nodes[n]
                if self.node_to_label:
                    new_nodes_to_label[new_n] = self.node_to_label[n]
                if self.node_to_id:
                    new_nodes_to_id[new_n] = self.node_to_id[n]
                new_nodes.append(new_n)
                new_node_presence.append([t0, t1])

        new_links = []
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if u in nodes_to_new_nodes and v in nodes_to_new_nodes:
                new_u = nodes_to_new_nodes[u]
                new_v = nodes_to_new_nodes[v]
                lu, lv = [], []
                for t0, t1 in zip(lp[::2], lp[1::2]):
                    tu, tv = None, None
                    for a, b in zip(new_node_presence[new_u][::2], new_node_presence[new_u][1::2]):
                        if a <= t1 and t0 <= b:  # Intersection
                            tu = (max(a, t0), min(t1, b))
                            break
                    for a, b in zip(new_node_presence[new_v][::2], new_node_presence[new_v][1::2]):
                        if a <= t1 and t0 <= b:  # Intersection
                            tv = (max(a, t0), min(t1, b))
                            break
                    if tu and tv:
                        lu.append(tu)
                        lv.append(tv)
                if lu and lv:
                    new_links.append((new_u, new_v))
                    uv_presence = []
                    for tu, tv in zip(lu, lv):
                        uv_presence += [max(tu[0], tv[0]), min(tv[1], tv[1])]
                    new_link_presence.append(uv_presence)

        S = StreamGraph(id=substream_id,
                        times=self.times,
                        nodes=new_nodes,
                        node_to_label=new_nodes_to_label,
                        node_to_id=new_nodes_to_id,
                        node_presence=new_node_presence,
                        links=new_links,
                        link_presence=new_link_presence)
        if return_new_node_label:
            return S, nodes_to_new_nodes
        return S

    ############################################################################################
    #          FILTERS( dot not use without reading doc, intended for StraphViz)               #
    ############################################################################################

    def filter_by_time_window(self, a, b):
        """
        Return the sub stream induced by T = [a,b].

        :param a:
        :param b:
        :return:
        """
        new_node_presence = []
        new_link_presence = []
        for n, np in zip(self.nodes, self.node_presence):
            new_np = []
            for t0, t1 in zip(np[::2], np[1::2]):
                if t0 <= b and a <= t1:  # Intersection
                    new_np += [max(a, t0), min(b, t1)]
            # if new_np:
            new_node_presence.append(new_np)

        for l, lp in zip(self.links, self.link_presence):
            new_lp = []
            for t0, t1 in zip(lp[::2], lp[1::2]):
                if t0 <= b and a <= t1:
                    new_lp += [max(a, t0), min(b, t1)]
            # if new_lp:
            new_link_presence.append(new_lp)

        return StreamGraph(times=[a, b], nodes=self.nodes,
                           node_to_label=self.node_to_label,
                           node_to_id=self.node_to_id,
                           node_presence=new_node_presence,
                           links=self.links,
                           link_presence=new_link_presence)

    def filter_by_links(self, links_list):
        """
        Return the sub stream induced by links in *links_list* and nodes implicated in these links.

        :param links_list: list of links to filter by
        :return:
        """
        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        links_target = [(js['source'], js['target']) for js in links_list]
        new_links = []
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if (u, v) in links_target or (v, u) in links_target:
                if u not in nodes_to_new_nodes:
                    new_u = nodes_to_new_nodes[u]
                    new_nodes_to_label[new_u] = self.node_to_label[u]
                    new_nodes_to_id[new_u] = self.node_to_id[u]
                    new_nodes.append(new_u)
                    new_node_presence.append(self.node_presence[u])
                else:
                    new_u = nodes_to_new_nodes[u]
                if v not in nodes_to_new_nodes:
                    new_v = nodes_to_new_nodes[v]
                    new_nodes_to_label[new_v] = self.node_to_label[v]
                    new_nodes_to_id[new_v] = self.node_to_id[v]
                    new_nodes.append(new_v)
                    new_node_presence.append(self.node_presence[v])
                else:
                    new_v = nodes_to_new_nodes[v]
                new_links.append((new_u, new_v))
                new_link_presence.append(lp)
        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def filter_by_links_id(self, links_list):
        """
        Return the sub stream induced by links in *links_list* and nodes implicated in these links.

        :param links_list: list of links to filter by
        :return:
        """
        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        links_target = [(int(js['source']), int(js['target'])) for js in links_list]
        new_links = []
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if (self.node_to_id[u], self.node_to_id[v]) in links_target \
                    or (self.node_to_id[v], self.node_to_id[u]) in links_target:
                if u not in nodes_to_new_nodes:
                    new_u = nodes_to_new_nodes[u]
                    new_nodes_to_label[new_u] = self.node_to_label[u]
                    new_nodes_to_id[new_u] = self.node_to_id[u]
                    new_nodes.append(new_u)
                    new_node_presence.append(self.node_presence[u])
                else:
                    new_u = nodes_to_new_nodes[u]
                if v not in nodes_to_new_nodes:
                    new_v = nodes_to_new_nodes[v]
                    new_nodes_to_label[new_v] = self.node_to_label[v]
                    new_nodes_to_id[new_v] = self.node_to_id[v]
                    new_nodes.append(new_v)
                    new_node_presence.append(self.node_presence[v])
                else:
                    new_v = nodes_to_new_nodes[v]
                new_links.append((new_u, new_v))
                new_link_presence.append(lp)
        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def filter_by_links_label(self, links_list):
        """
        Return the sub stream induced by links in *links_list* and nodes implicated in these links.

        :param links_list: list of links to filter by
        :return:
        """
        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        new_links = []
        links_target = [(js['source'], js['target']) for js in links_list]
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if (self.node_to_label[u], self.node_to_label[v]) in links_target or \
                    (self.node_to_label[v], self.node_to_label[u]) in links_target:
                if u not in nodes_to_new_nodes:
                    new_u = nodes_to_new_nodes[u]
                    new_nodes_to_label[new_u] = self.node_to_label[u]
                    new_nodes_to_id[new_u] = self.node_to_id[u]
                    new_nodes.append(new_u)
                    new_node_presence.append(self.node_presence[u])
                else:
                    new_u = nodes_to_new_nodes[u]
                if v not in nodes_to_new_nodes:
                    new_v = nodes_to_new_nodes[v]
                    new_nodes_to_label[new_v] = self.node_to_label[v]
                    new_nodes_to_id[new_v] = self.node_to_id[v]
                    new_nodes.append(new_v)
                    new_node_presence.append(self.node_presence[v])
                else:
                    new_v = nodes_to_new_nodes[v]
                new_links.append((new_u, new_v))
                new_link_presence.append(lp)
        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def filter_by_nodes(self, nodes_list):
        """
        Return the sub stream graph induced by nodes in *nodes_target*.
        If an extremity of a link is in *nodes_list* we add the other node.
        Different from induced stream graph

        :param nodes_list:
        :return:
        """
        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        for n, np in zip(self.nodes, self.node_presence):
            if n in nodes_list:
                new_n = nodes_to_new_nodes[n]
                new_nodes_to_label[new_n] = self.node_to_label[n]
                if self.node_to_id:
                    new_nodes_to_id[new_n] = self.node_to_id[n]
                new_nodes.append(new_n)
                new_node_presence.append(np)

        new_links = []
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if u in nodes_list or v in nodes_list:
                if u not in nodes_to_new_nodes:
                    new_u = nodes_to_new_nodes[u]
                    new_nodes_to_label[new_u] = self.node_to_label[u]
                    if self.node_to_id:
                        new_nodes_to_id[new_u] = self.node_to_id[u]
                else:
                    new_u = nodes_to_new_nodes[u]

                if v not in nodes_to_new_nodes:
                    new_v = nodes_to_new_nodes[v]
                    new_nodes_to_label[new_v] = self.node_to_label[v]
                    if self.node_to_id:
                        new_nodes_to_id[new_v] = self.node_to_id[v]

                else:
                    new_v = nodes_to_new_nodes[v]

                new_links.append((new_u, new_v))
                new_link_presence.append(lp)

        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def filter_by_node_id(self, nodes_target):
        """
        Return the sub stream graph induced by nodes in *nodes_target*.
        Different from induced stream graph (that include nodes in links containing a node in nodes_list)

        :param nodes_target:
        :return:
        """
        nodes_target = [int(n) for n in nodes_target]
        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        for n, np in zip(self.nodes, self.node_presence):
            if self.node_to_id[n] in nodes_target:
                new_n = nodes_to_new_nodes[n]
                new_nodes_to_label[new_n] = self.node_to_label[n]
                new_nodes_to_id[new_n] = self.node_to_id[n]
                new_nodes.append(new_n)
                new_node_presence.append(np)

        new_links = []
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if self.node_to_id[u] in nodes_target or self.node_to_id[v] in nodes_target:
                if u not in nodes_to_new_nodes:
                    new_u = nodes_to_new_nodes[u]
                    new_nodes_to_label[new_u] = self.node_to_label[u]
                    new_nodes_to_id[new_u] = self.node_to_id[u]
                else:
                    new_u = nodes_to_new_nodes[u]

                if v not in nodes_to_new_nodes:
                    new_v = nodes_to_new_nodes[v]
                    new_nodes_to_label[new_v] = self.node_to_label[v]
                    new_nodes_to_id[new_v] = self.node_to_id[v]

                else:
                    new_v = nodes_to_new_nodes[v]

                new_links.append((new_u, new_v))
                new_link_presence.append(lp)

        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    def filter_by_node_label(self, nodes_target):
        """
        Return the sub stream graph induced by nodes in *nodes_target*.

        :param nodes_target:
        :return:
        """
        new_nodes = []
        new_node_presence = []
        new_nodes_to_label = {}
        new_nodes_to_id = {}
        nodes_to_new_nodes = defaultdict(lambda: len(nodes_to_new_nodes))
        for n, np in zip(self.nodes, self.node_presence):
            if self.node_to_label[n] in nodes_target:
                new_n = nodes_to_new_nodes[n]
                new_nodes_to_label[new_n] = self.node_to_label[n]
                new_nodes_to_id[new_n] = self.node_to_id[n]
                new_nodes.append(new_n)
                new_node_presence.append(np)

        new_links = []
        new_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            u, v = l
            if self.node_to_label[u] in nodes_target or self.node_to_label[v] in nodes_target:
                if u not in nodes_to_new_nodes:
                    new_u = nodes_to_new_nodes[u]
                    new_nodes_to_label[new_u] = self.node_to_label[u]
                    new_nodes_to_id[new_u] = self.node_to_id[u]
                    # new_nodes.append(new_u)
                    # new_node_presence.append([])
                else:
                    new_u = nodes_to_new_nodes[u]

                if v not in nodes_to_new_nodes:
                    new_v = nodes_to_new_nodes[v]
                    new_nodes_to_label[new_v] = self.node_to_label[v]
                    new_nodes_to_id[new_v] = self.node_to_id[v]
                    # new_nodes.append(new_v)
                    # new_node_presence.append([])
                else:
                    new_v = nodes_to_new_nodes[v]

                new_links.append((new_u, new_v))
                new_link_presence.append(lp)

        return StreamGraph(times=self.times,
                           nodes=new_nodes,
                           node_to_label=new_nodes_to_label,
                           node_to_id=new_nodes_to_id,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    ##########################################################################
    #               Snapshots                                                #
    ##########################################################################

    def to_snapshots(self):
        G_list = []
        E = self.ordered_events()
        current_a_l = {}
        t_old = self.times[0]
        last_c = E[0][0]
        for ev in E:
            c, t = ev[0], ev[1]
            if t != t_old:
                G_list.append((t_old, t, {k: list(v) for k, v in current_a_l.items()}))
            if c == 1:
                (b, e, u, v) = ev[1:]
                current_a_l[u].add(v)
                current_a_l[v].add(u)
            elif c == -1:
                if t == t_old and last_c > 0:
                    G_list.append((t_old, t, {k: list(v) for k, v in current_a_l.items()}))
                (e, u, v) = ev[1:]
                current_a_l[u].remove(v)
                current_a_l[v].remove(u)
            elif c == 2:
                (b, e, u) = ev[1:]
                current_a_l[u] = set()
            elif c == -2:
                if t == t_old and last_c > 0:
                    G_list.append((t_old, t, {k: list(v) for k, v in current_a_l.items()}))
                (e, u) = ev[1:]
                current_a_l.pop(u)
            t_old = t
            last_c = c
        return G_list

    def nb_cc_from_adjacency_list(self, a_l):
        nb_components = 0
        unvisited = set(a_l.keys())
        while len(unvisited) > 0:
            v = unvisited.pop()
            current_component, visited = DFS_iterative(v, a_l)
            unvisited -= visited
            nb_components += 1
        return nb_components

    def number_of_connected_components_in_snapshots(self):
        cnt_cc = 0
        E = self.ordered_events()
        current_a_l = {}
        t_old = self.times[0]
        last_c = E[0][0]
        for ev in E:
            c, t = ev[0], ev[1]
            if t != t_old:
                cnt_cc += self.nb_cc_from_adjacency_list(current_a_l)
            if c == 1:
                (b, e, u, v) = ev[1:]
                current_a_l[u].add(v)
                current_a_l[v].add(u)
            elif c == -1:
                if t == t_old and last_c > 0:
                    cnt_cc += self.nb_cc_from_adjacency_list(current_a_l)
                (e, u, v) = ev[1:]
                current_a_l[u].remove(v)
                current_a_l[v].remove(u)
            elif c == 2:
                (b, e, u) = ev[1:]
                current_a_l[u] = set()
            elif c == -2:
                if t == t_old and last_c > 0:
                    cnt_cc += self.nb_cc_from_adjacency_list(current_a_l)
                (e, u) = ev[1:]
                current_a_l.pop(u)
            t_old = t
            last_c = c
        return cnt_cc

    ##########################################################################
    #               Stable Connected Components                              #
    ##########################################################################

    def stable_connected_components(self, format="object_with_links", node_list=None):
        """

        :param node_list:
        :param format: "cluster" to obtain StCC under the form of [(t0,t1,u),...]
                        "object" to obtain StCC under the form of objects of class "connected_components"
        :return:
        """
        return cmp.compute_stable_connected_components(self, format=format, node_list=node_list)

    def number_of_stable_connected_component(self, format="cluster"):
        """
        Return the number of strongly connected components in the stream graph.

        :return:
        """
        return len(self.stable_connected_components(format=format))

    ##########################################################################
    #               Weakly Connected Components                            #
    ##########################################################################

    def weakly_connected_components(self, streaming=False, reformat=False, free_memory=False):
        """
        Compute the weakly connected components of the Stream Graph.
        Time Complexity in $O(M+N)$.

        :param streaming:
        :param reformat:
        :param free_memory:
        :return: Weakly Connected Components of the Stream Graph (a list of clusters : [[(t0,t1,u),...],[(t0,t1,v)]])
        """
        if streaming:
            return cmp.compute_wcc_streaming(self, reformat=reformat, free_memory=free_memory)

        else:
            return cmp.compute_wcc_dfs(self, free_memory=free_memory)

    def number_of_weakly_connected_component(self):
        """
        Return the number of weakly connected components in the stream graph.

        :return:
        """
        return len(self.weakly_connected_components())

    def wcc_as_substreams(self, n_jobs=-1):
        """
        Return the weakly connected components of the stream graph
        in the form of substreams (stream graph induced by the component/cluster)

        :param n_jobs: Number of cores to allocate for a parallel computation
        :return:
        """

        return cmp.compute_wcc_as_substreams(self, n_jobs)

    ##########################################################################
    #               Strongly Connected Components                            #
    ##########################################################################

    def strongly_connected_components(self, format="object", streaming_output=None,
                                      method="Direct", isolated_nodes=True, free_memory=False):
        """
        Compute the strongly connected components of the Stream Graph
        
        :param free_memory:
        :param isolated_nodes:
        :param method:
        :param streaming_output:
        :param format: "cluster" to obtain SCC under the form of [(t0,t1,u),...]
                        "object" to obtain SCC under the form of objects of class "strongly_connected_components"
        :return:
        """
        if method == "Direct":
            return cmp.compute_strongly_connected_components(self, format=format,
                                                             streaming_output=streaming_output,
                                                             isolated_nodes=isolated_nodes,
                                                             free_memory=free_memory)
        elif method == "UF":
            return cmp.strongly_connected_components_UF(self, format=format)

        elif method == "FD":
            return etf.strongly_connected_components_ETF(self, format=format,
                                                         streaming_output=streaming_output,
                                                         isolated_nodes=isolated_nodes,
                                                         free_memory=free_memory)
        else:
            raise ValueError("Method " + method + " is not supported for the computation of SCC. Supported methods"
                                                  "are 'Direct', 'UF' and 'FD'.")

    def number_of_strongly_connected_component(self, format="cluster"):
        """
        Return the number of strongly connected components in the stream graph.

        :return:
        """
        return len(self.strongly_connected_components(format=format))

    ############################################################
    #                     Condensation DAG                     #
    ############################################################

    def condensation_dag(self, format="object_with_links", free_memory=False, isolated_nodes=True):
        """
        Compute the Condensation Graph of the Stream Graph.
        
        :return: Condensation Graph
        """

        _, dag = cmp.compute_strongly_connected_components(self, condensation_dag=True, format=format,
                                                           free_memory=free_memory, isolated_nodes=isolated_nodes)
        dag.set_id_comp_to_comp()

        if format == "cluster":
            dag.times = (min([c[0][0] for c in dag.c_nodes]), max([c[0][1] for c in dag.c_nodes]))
            dag.compute_links_inplace()  # Add Links

        elif format == "object" or format == "object_with_links":
            dag.times = (min([c.times[0] for c in dag.c_nodes]), max([c.times[1] for c in dag.c_nodes]))
            dag.set_index_node_to_id_comp()
            dag.compute_links_inplace()  # Add Links
            if format == "object_with_links":
                dag.set_index_segmented_node_to_id_comp([(t0, t1, n)
                                                         for n, np in zip(self.nodes, self.node_presence)
                                                         for t0, t1 in zip(np[::2], np[1::2])])
        # TODO: Add test
        # adj_list = dag.adjacency_list()
        # G = nx.from_dict_of_lists(adj_list,nx.DiGraph)
        # print(nx.find_cycle(G))
        return dag

    def stable_dag(self, format="object_with_links", free_memory=False, isolated_nodes=True,
                   node_list=None):
        """
        Compute the Stable DAG of the Stream Graph.

        :param format:
        :param free_memory:
        :param isolated_nodes:
        :param node_list:
        :return:
        """
        _, dag = cmp.compute_stable_connected_components(self, stable_dag=True, format=format,
                                                         free_memory=free_memory, isolated_nodes=isolated_nodes,
                                                         node_list=node_list)
        dag.set_id_comp_to_comp()

        if format == "cluster":
            dag.times = (min([c[0][0] for c in dag.c_nodes]), max([c[0][1] for c in dag.c_nodes]))
            dag.compute_links_inplace()  # Add Links

        elif format == "object" or format == "object_with_links":
            dag.times = (min([c.times[0] for c in dag.c_nodes]), max([c.times[1] for c in dag.c_nodes]))
            dag.set_index_node_to_id_comp()
            dag.compute_links_inplace()  # Add Links
            if format == "object_with_links":
                dag.set_index_segmented_node_to_id_comp([(t0, t1, n)
                                                         for n, np in zip(self.nodes, self.node_presence)
                                                         for t0, t1 in zip(np[::2], np[1::2])])

        # TODO: Add test
        # adj_list = dag.adjacency_list()
        # G = nx.from_dict_of_lists(adj_list,nx.DiGraph)
        # print(nx.find_cycle(G))
        return dag

    ############################################################
    #               Kcores                                     #
    ############################################################
    def k_core(self, k, stable_dag=None, n_jobs=-1):
        """
        Compute the *k*-core cluster of the Stream Graph.

        :param n_jobs:
        :param k: *k*-core
        :param stable_dag: CondensationDag: Condensation DAG (optional)
        :return:
        """
        if stable_dag is None:
            stable_dag = self.stable_dag()
        k_core = stable_dag.k_core(k, n_jobs=n_jobs)
        return k_core

    def core_number(self, stable_dag=None, n_jobs=-1):
        """
        Compute the core number of each temporal node.

        :param n_jobs:
        :param stable_dag: Stable DAG (optional)
        :return:
        """
        if stable_dag is None:
            stable_dag = self.stable_dag()
        core_number = stable_dag.core_number(n_jobs=n_jobs)
        isolated_nodes = self.isolated_nodes()
        core_number[0] = isolated_nodes
        return core_number

    def average_core_size(self, n_jobs=-1):
        kcores = self.core_number(n_jobs=n_jobs)
        T = self.times[1] - self.times[0]
        average_core_size = sum([k * sum([n[1] - n[0] for n in v])
                                 for k, v in kcores.items()]) / (self.nb_nodes() * T)
        return average_core_size

    ############################################################
    #               Kcliques                                   #
    ############################################################

    def k_clique(self, k, stable_dag=None, n_jobs=-1):
        """
        Compute the *k*-cliques of the Stream Graph

        :param n_jobs:
        :param k:
        :param stable_dag:
        :return:
        """
        if stable_dag is None:
            stable_dag = self.stable_dag()
        k_clique = stable_dag.k_clique(k, n_jobs=n_jobs)
        return k_clique

    def all_cliques(self, stable_dag=None, n_jobs=-1, format="dict"):
        """
        Return all the cliques of the Stream Graph

        :param format:
        :param format:
        :param n_jobs:
        :param stable_dag:
        :return:
        """
        if stable_dag is None:
            stable_dag = self.stable_dag()
        all_cliques = stable_dag.all_cliques(n_jobs=n_jobs)
        if format == "cluster":
            return postprocess_kcliques_into_clusters(all_cliques)
        return all_cliques

    def max_clique_number(self, stable_dag=None, n_jobs=-1):
        """
        Return the size of the maximal clique for each node (in the form of temporal windows : a cluster)

        :return:
        """
        print("NOT IMPLEMENTED YET !!")
        if stable_dag is None:
            stable_dag = self.stable_dag()
        all_cliques = stable_dag.all_cliques(n_jobs=n_jobs)
        max_clique = {v: [] for v in self.nodes}
        for k, cluster in all_cliques.items():
            for t0, t1, n in cluster:
                max_clique[n] = [t0, t1, n]
        # TODO: To finish...
        return

    def average_clique_size(self, n_jobs=-1):
        kcliques = self.all_cliques(n_jobs=n_jobs)
        T = self.times[1] - self.times[0]
        average_clique_size = sum([k * sum([n[1] - n[0] for i in v for n in i])
                                   for k, v in kcliques.items()]) / (self.nb_nodes() * T)
        return average_clique_size

    ###########################################################
    #                   Degree Partition                      #
    ###########################################################

    def neighborhood_node_2_link_presence(self):
        N = {(t0, t1, n): []
             for n, np in zip(self.nodes, self.node_presence)
             for t0, t1 in zip(np[::2], np[1::2])}
        for l, lp in zip(self.links, self.link_presence):
            for t0, t1 in zip(lp[::2], lp[1::2]):
                np1, np2 = self.get_according_node_presence_from_link(l, t0, t1)
                N[np1] += [t0, t1]
                N[np2] += [t0, t1]
        return N

    def degrees_partition(self, E=None):

        node_degree = defaultdict(list)  # (operation code, time , degree)
        if E is None:
            E = self.ordered_events()

        for e in E:
            c = e[0]

            if c == 2:
                _, t0, t1, n = e
                node_degree[n].append((c, t0, 0))
            elif c == 1:
                _, t0, t1, u, v = e
                pred_c, pred_t, pred_d = node_degree[u][-1]
                if pred_t == t0 and pred_c in {1, 2}:
                    node_degree[u][-1] = (c, t0, pred_d + 1)
                else:
                    node_degree[u].append((c, t0, pred_d))
                    node_degree[u].append((c, t0, pred_d + 1))

                pred_c, pred_t, pred_d = node_degree[v][-1]
                if pred_t == t0 and pred_c in {1, 2}:
                    node_degree[v][-1] = (c, t0, pred_d + 1)
                else:
                    node_degree[v].append((c, t0, pred_d))
                    node_degree[v].append((c, t0, pred_d + 1))

            elif c == -1:
                _, t1, u, v = e

                pred_c, pred_t, pred_d = node_degree[u][-1]
                if pred_t == t1 and pred_c == -1:
                    node_degree[u][-1] = (c, t1, pred_d - 1)
                else:
                    node_degree[u].append((c, t1, pred_d))
                    node_degree[u].append((c, t1, pred_d - 1))

                pred_c, pred_t, pred_d = node_degree[v][-1]
                if pred_t == t1 and pred_c == -1:
                    node_degree[v][-1] = (c, t1, pred_d - 1)
                else:
                    node_degree[v].append((c, t1, pred_d))
                    node_degree[v].append((c, t1, pred_d - 1))

            elif c == -2:
                _, t1, n = e
                pred_c, pred_t, pred_d = node_degree[n][-1]
                if pred_t == t1 and pred_c in {-1, -2}:
                    del node_degree[n][-1]
                else:
                    node_degree[n].append((c, t1, 0))

        # PostProcess into dict clusters
        degrees = defaultdict(set)
        for n, seq in node_degree.items():
            for (_, t0, d0), (_, t1, d1) in zip(seq[::2], seq[1::2]):
                degrees[d0].add((t0, t1, n))
        return degrees

    def get_interaction_times(self):
        """
        O(M)

        :return:
        """
        interaction_times = defaultdict(list)
        for l, lp in zip(self.links, self.link_presence):
            interaction_times[l[0]] += lp
            interaction_times[l[1]] += lp
        return interaction_times

    def nodes_partition_on_degrees(self, interaction_times=None):
        if not interaction_times:
            interaction_times = self.get_interaction_times()
        nodes_partition = {n: [] for n in self.nodes}
        for n in interaction_times:
            t_n = interaction_times[n]
            if t_n:
                weights = [0] * len(t_n)
                l = len(t_n) // 2
                weights[::2] = [1] * l
                weights[1::2] = [-1] * l
                weights = [x for _, x in sorted(zip(t_n, weights))]
                t_n = sorted(t_n)
                cum_weights = list(itertools.accumulate(weights))
                for i in range(len(weights)):
                    if cum_weights[i] == 0:
                        continue
                    if t_n[i] != t_n[i + 1]:
                        nodes_partition[n] += [t_n[i], t_n[i + 1]]
        return nodes_partition

    def nodes_degree(self, interaction_times=None):
        if not interaction_times:
            interaction_times = self.get_interaction_times()
        nodes_degree = {n: [] for n in self.nodes}
        for n in interaction_times:
            t_n = interaction_times[n]
            if t_n:
                weights = [0] * len(t_n)
                l = len(t_n) // 2
                weights[::2] = [1] * l
                weights[1::2] = [-1] * l
                weights = [x for _, x in sorted(zip(t_n, weights))]
                t_n = sorted(t_n)
                cum_weights = list(itertools.accumulate(weights))
                for i in range(len(weights)):
                    if cum_weights[i] == 0:
                        continue
                    if t_n[i] != t_n[i + 1]:
                        nodes_degree[n] += [(t_n[i], t_n[i + 1], cum_weights[i])]
        return nodes_degree

    ###################################
    #             PATHS               #
    ###################################

    def times_to_reach(self, source=None, destination=None, start_time=None, enumerate=False, E=None):
        """
        Compute temporal paths in the Stream Graph.
        If both *source* and *destination* are specified: return the time to reach of the foremost path
        If only *source* is specified: return time to reach all other nodes (single source)
        If *source* and *destination* aren't specified: return times to reach pairwise.

        :param E:
        :param start_time:
        :param source: Can be a source node or a temporal source node(t_0,t_1,u)
        :param destination: Can be a destination node or a temporal destination node(t_0,t_1,v)
        :param enumerate: if True we enumerate the number of path
        :return: Times to reach
        """

        if type(source) is str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            source = label_to_node[source]
        if source is not None:
            return ap.FoP(self, source, destination, start_time, E=E)
        else:
            raise ValueError("Pairwise algorithm for path computation is not yet available."
                             "Must past a 'source' paramater.")
            # return ap.FoP_pw(self)

    def times_to_reach_and_lengths(self, source=None, destination=None, start_time=None, enumerate=False, E=None):
        """
        Compute temporal paths in the Stream Graph.
        If both *source* and *destination* are specified: return the ttr and distance of the shortest foremost path
        If only *source* is specified: return ttr and distances to all other nodes (single source)
        If *source* and *destination* aren't specified: return ttr and distances pairwise.

        :param E:
        :param start_time:
        :param source: Can be a source node or a temporal source node(t_0,t_1,u)
        :param destination: Can be a destination node or a temporal destination node(t_0,t_1,v)
        :param enumerate: if True we enumerate the number of path
        :return: Times to reach and distances
        """
        if type(source) is str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            source = label_to_node[source]

        if source is not None:
            return ap.SFoP(self, source, destination, start_time, E=E)
        else:
            raise ValueError("Pairwise algorithm for path computation is not yet available."
                             "Must past a 'source' paramater.")
            # return ap.SFoP_pw(self)

    def latencies(self, source=None, destination=None, enumerate=False, E=None):
        """
        Compute temporal paths in the Stream Graph.
        If both *source* and *destination* are specified: return the latency of the fastest path
        If only *source* is specified: return latencies to all other nodes (single source)
        If *source* and *destination* aren't specified: return latencies pairwise.

        :param E:
        :param source: Can be a source node or a temporal source node(t_0,t_1,u)
        :param destination: Can be a destination node or a temporal destination node(t_0,t_1,v)
        :param enumerate: if True we enumerate the number of path
        :return: Latencies
        """
        if type(source) is str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            source = label_to_node[source]

        if source is not None:
            return ap.FP(self, source, destination, E=E)
        else:
            raise ValueError("Pairwise algorithm for path computation is not yet available."
                             "Must past a 'source' paramater.")
            # return ap.FP_pw(self)

    def latencies_and_lengths(self, source=None, destination=None, enumerate=False, E=None):
        """
        Compute temporal paths in the Stream Graph.
        If both *source* and *destination* are specified: return the latency_and_length of the shortest fastest path
        If only *source* is specified: return latencies and lengths to all other nodes (single source)
        If *source* and *destination* aren't specified: return latencies and lengths pairwise.

        :param E:
        :param source: Can be a source node or a temporal source node(t_0,t_1,u)
        :param destination: Can be a destination node or a temporal destination node(t_0,t_1,v)
        :param enumerate: if True we enumerate the number of path
        :return: Latencies and lengths
        """
        if type(source) is str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            source = label_to_node[source]

        if source is not None:
            return ap. \
                SFP(self, source, destination, E=E)
        else:
            raise ValueError("Pairwise algorithm for path computation is not yet available."
                             "Must past a 'source' paramater.")
            # return ap.SFP_pw(self)

    def distances(self, source=None, destination=None, enumerate=False, E=None):
        """
        Compute temporal paths in the Stream Graph.
        If both *source* and *destination* are specified: return the distance of the shortest path
        If only *source* is specified: return distances to all other nodes (single source)
        If *source* and *destination* aren't specified: return distances pairwise.

        :param E:
        :param source: Can be a source node or a temporal source node(t_0,t_1,u)
        :param destination: Can be a destination node or a temporal destination node(t_0,t_1,v)
        :param enumerate: if True we enumerate the number of path
        :return: Distances
        """
        if type(source) is str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            source = label_to_node[source]

        if source is not None:
            return ap.SP(self, source, destination, E=E)
        else:
            raise ValueError("Pairwise algorithm for path computation is not yet available."
                             "Must past a 'source' paramater.")
            # return ap.SP_pw(self)

    def distances_and_durations(self, source=None, destination=None, enumerate=False, E=None):
        """
        Compute temporal paths in the Stream Graph.
        If both *source* and *destination* are specified: return the distance and duration of the fastest shortest path
        If only *source* is specified: return distances and durations to all other nodes (single source)
        If *source* and *destination* aren't specified: return distances and durations pairwise.

        :param E:
        :param source: Can be a source node or a temporal source node(t_0,t_1,u)
        :param destination: Can be a destination node or a temporal destination node(t_0,t_1,v)
        :param enumerate: if True we enumerate the number of path
        :return: Distances and durations
        """
        if type(source) is str:
            label_to_node = {v: k for k, v in self.node_to_label.items()}
            source = label_to_node[source]

        if source is not None:
            return ap.FSP(self, source, destination, E=E)
        else:
            raise ValueError("Pairwise algorithm for path computation is not yet available."
                             "Must past a 'source' paramater.")
            # return ap.FSP_pw(self)

    #############################################################
    #               Properties
    #############################################################

    # def links_properties(self, list_properties, to_pandas=False):
    #     # TODO : TO FINISH
    #     """
    # 
    #     :param list_properties:  properties of the Stream Graph
    #     :param to_pandas: If 'True' return a pandas DataFrame
    #     :return:
    #     """
    #     property_to_method = {'duration': self.duration,
    #                           'surface': self.get_card_E,
    #                           'stream_graph_degree': self.stream_graph_degree,
    #                           'expected_stream_graph_degree': self.expected_stream_graph_degree,
    #                           'average_degree': self.average_degree,
    #                           'nb_nodes': self.nb_nodes,
    #                           'nb_links': self.nb_links,
    #                           'node_duration': self.node_duration,
    #                           'link_duration': self.link_duration,
    #                           'coverage': self.coverage,
    #                           'density': self.density,
    #                           'average_clustering_coefficient': self.average_clustering,
    #                           'nb_of_wcc': self.number_of_weakly_connected_component,
    #                           'nb_of_scc': self.number_of_strongly_connected_component,
    #                           'average_core_size': self.average_core_size,
    #                           'average_clique_size': self.average_clique_size,
    #                           # 'uniformity':self.uniformity, Too costly !!
    #                           }
    #     default_properties = {'n': len(self.nodes),
    #                           'N': sum([len(item) / 2 for item in self.node_presence]),
    #                           'm': len(self.links),
    #                           'M': sum([len(item) / 2 for item in self.link_presence]),
    #                           }
    # 
    #     if list_properties == "all":
    #         dict_data = {p: [] for p in property_to_method}
    #         for p in property_to_method:
    #             chrono = time.time()
    #             dict_data[p].append(property_to_method[p]())
    #             print("Time to compute ", p, " : ", time.time() - chrono)
    #     elif list_properties != "default":
    #         dict_data = {p: [] for p in list_properties}
    #         for p in list_properties:
    #             dict_data[p].append(property_to_method[p]())
    # 
    #     for p in default_properties:
    #         dict_data[p] = [default_properties[p]]
    # 
    #     if to_pandas:
    #         if self.id:
    #             Index = pd.Index([self.id])
    #         else:
    #             Index = pd.Index(['Stream Graph'])
    #         D = pd.DataFrame(dict_data, index=Index)
    #         return D
    # 
    #     return dict_data

    def nodes_max_degree(self):
        max_degree = Counter()
        nodes_degree = self.nodes_degree()
        for n, d in nodes_degree.items():
            max_degree[n] = max(d)
        return max_degree

    def nodes_average_core_number(self, core_number=None):
        nodes_duration = self.node_duration()
        nodes_average_core_number = {n: 0 for n in self.nodes}
        if core_number is None:
            core_number = self.core_number()
        for c in core_number:
            for n in core_number[c]:
                t0, t1, u = n
                nodes_average_core_number[u] += (t1 - t0) * c
        for u in nodes_average_core_number:
            #  TODO : NODES DURATION
            nodes_average_core_number[u] = nodes_average_core_number[u] / nodes_duration[u]
        return

    # def nodes_properties(self, list_properties, to_pandas=False):
    #     """
    #     # TODO : TO FINISH
    #     :param list_properties:  properties of the Stream Graph
    #     :param to_pandas: If 'True' return a pandas DataFrame
    #     :return:
    #     """
    #     property_to_method = {'degree': self.degrees,
    #                           'max_degree': self.nodes_max_degree,
    #                           'average_core_number': self.nodes_average_core_number,
    #                           'max_core_number': self.nodes_max_core_number,
    #                           'node_duration': self.node_duration,
    #                           'average_clique_number': self.nodes_average_clique_number,
    #                           'max_clique_number': self.nodes_max_clique_number,
    #                           'clustering_coefficient': self.clustering_coefficient,
    #                           'mean_degree_neighborhood': self.mean_degree_neighborhood,
    #                           'expected_node_degree': self.expected_node_degree,
    #                           'average_size_wcc': self.nodes_average_size_wcc,
    #                           'average_size_scc': self.nodes_average_size_scc,
    #                           }
    # 
    #     default_properties = {'n': len(self.nodes),
    #                           'N': sum([len(item) / 2 for item in self.node_presence]),
    #                           'm': len(self.links),
    #                           'M': sum([len(item) / 2 for item in self.link_presence]),
    #                           }
    # 
    #     if list_properties == "all":
    #         dict_data = {p: [] for p in property_to_method}
    #         for p in property_to_method:
    #             chrono = time.time()
    #             dict_data[p].append(property_to_method[p]())
    #             print("Time to compute ", p, " : ", time.time() - chrono)
    #     elif list_properties != "default":
    #         dict_data = {p: [] for p in list_properties}
    #         for p in list_properties:
    #             dict_data[p].append(property_to_method[p]())
    # 
    #     for p in default_properties:
    #         dict_data[p] = [default_properties[p]]
    # 
    #     if to_pandas:
    #         if self.id:
    #             Index = pd.Index([self.id])
    #         else:
    #             Index = pd.Index(['Stream Graph'])
    #         D = pd.DataFrame(dict_data, index=Index)
    #         return D
    # 
    #     return dict_data

    # def properties(self, list_properties="default", to_pandas=False):
    #     """
    # 
    #     :param list_properties:  properties of the Stream Graph
    #     :param to_pandas: If 'True' return a pandas DataFrame
    #     :return:
    #     """
    #     property_to_method = {'duration': self.duration,
    #                           'stream_graph_degree': self.stream_graph_degree,
    #                           'expected_stream_graph_degree': self.expected_stream_graph_degree,
    #                           'average_degree': self.average_degree,
    #                           'nb_nodes': self.nb_nodes,
    #                           'nb_links': self.nb_links,
    #                           'node_duration': self.node_duration,
    #                           'link_duration': self.link_duration,
    #                           'coverage': self.coverage,
    #                           'density': self.density,
    #                           'average_clustering_coefficient': self.average_clustering,
    #                           'nb_of_wcc': self.number_of_weakly_connected_component,
    #                           'nb_of_scc': self.number_of_strongly_connected_component,
    #                           'average_core_size': self.average_core_size,
    #                           'average_clique_size': self.average_clique_size,
    #                           # 'uniformity':self.uniformity, Too costly !!
    #                           }
    #     default_properties = {'n': len(self.nodes),
    #                           'N': sum([len(item) / 2 for item in self.node_presence]),
    #                           'm': len(self.links),
    #                           'M': sum([len(item) / 2 for item in self.link_presence]),
    #                           }
    #     dict_data = {}
    #     if list_properties == "all":
    #         dict_data = {p: [] for p in property_to_method}
    #         for p in property_to_method:
    #             chrono = time.time()
    #             dict_data[p].append(property_to_method[p]())
    #             print("Time to compute ", p, " : ", time.time() - chrono)
    #     elif list_properties != "default":
    #         dict_data = {p: [] for p in list_properties}
    #         for p in list_properties:
    #             dict_data[p].append(property_to_method[p]())
    # 
    #     for p in default_properties:
    #         dict_data[p] = [default_properties[p]]
    # 
    #     if to_pandas:
    #         if self.id:
    #             Index = pd.Index([self.id])
    #         else:
    #             Index = pd.Index(['Stream Graph'])
    #         D = pd.DataFrame(dict_data, index=Index)
    #         return D
    # 
    #     return dict_data

    ############################################################
    #               Extend graph                               #
    ############################################################

    def extend_on_node_presence(self):
        # Split on nodes...

        New_nodes = [(n, t0, t1)
                     for n in self.nodes
                     for t0, t1 in zip(self.node_presence[n][::2],
                                       self.node_presence[n][1::2])
                     ]

        dict_node_2_id = {n: i for n, i in zip(New_nodes, range(len(New_nodes)))}
        New_links = []
        pos_link = {}
        New_link_presence = []
        for l, lp in zip(self.links, self.link_presence):
            for lt0, lt1 in zip(lp[::2], lp[1::2]):
                np1, np2 = self.get_according_node_presence_from_link(l, lt0, lt1)
                n1, n2 = dict_node_2_id[np1], dict_node_2_id[np2]
                if (np1, np2) not in New_links and (np2, np1) not in New_links:
                    pos_link[(n1, n2)] = len(New_links)
                    pos_link[(n2, n1)] = len(New_links)
                    New_links.append((n1, n2))
                    New_link_presence.append([lt0, lt1])
                else:
                    New_link_presence[pos_link[(n1, n2)]] += [lt0, lt1]

        return StreamGraph(nodes=[i for i in dict_node_2_id.values()],
                           node_presence=[[n[1], n[2]] for n in dict_node_2_id.keys()],
                           link_presence=New_link_presence,
                           links=New_links)

    def get_links_bounds_per_nodes(self, interaction_times=None):
        # Probably the best function to get bounds
        if not interaction_times:
            interaction_times = self.get_interaction_times()
        bounds = defaultdict(list)
        for n in interaction_times:
            t_n = interaction_times[n]
            if t_n:
                weights = [0] * len(t_n)
                l = len(t_n) // 2
                weights[::2] = [1] * l
                weights[1::2] = [-1] * l
                weights = [x for _, x in sorted(zip(t_n, weights))]
                t_n.sort()
                t0 = t_n[0]
                cnt = 0
                for i in range(len(weights)):
                    cnt_old = cnt
                    cnt += weights[i]
                    if cnt == 0:
                        bounds[n] += [t0, t_n[i]]
                    elif cnt_old == 0 and cnt == 1:
                        t0 = t_n[i]
        return bounds

    def extend_on_link_presence(self):
        # Split on links...
        new_nodes = []
        new_links = []
        new_node_presence = []
        new_link_presence = []
        interaction_times = {n: [] for n in self.nodes}
        for l, lp in zip(self.links, self.link_presence):
            interaction_times[l[0]] += lp
            interaction_times[l[1]] += lp
        nn = 0
        dict_nodes_2_new_nodes = {n: [] for n in self.nodes}
        for n in interaction_times:
            t_n = interaction_times[n]
            if not t_n:
                continue
            weights = numpy.empty(len(t_n))
            weights[::2] = 1
            weights[1::2] = -1
            ord = numpy.argsort(t_n)
            weights = weights[ord]
            weights = weights.cumsum()
            t_n.sort()
            t0 = t_n[0]

            for j in range(len(weights)):
                if weights[j] == 0:
                    dict_nodes_2_new_nodes[n].append(nn)
                    new_nodes.append(nn)
                    new_node_presence.append([t0, t_n[j]])
                    nn += 1
                elif weights[j - 1] == 0 and weights[j] == 1:
                    t0 = t_n[j]

        pos_link = {}
        nl1, nl2 = None, None
        for l, lp in zip(self.links, self.link_presence):
            # print("Old Link :",l,"Time :",lp)
            for lt0, lt1 in zip(lp[::2], lp[1::2]):
                for n1 in dict_nodes_2_new_nodes[l[0]]:
                    if new_node_presence[n1][0] <= lt0 and lt1 <= new_node_presence[n1][1]:
                        nl1 = n1
                        break
                for n2 in dict_nodes_2_new_nodes[l[1]]:
                    if new_node_presence[n2][0] <= lt0 and lt1 <= new_node_presence[n2][1]:
                        nl2 = n2
                        break
                if (nl1, nl2) not in new_links and (nl2, nl1) not in new_links:
                    pos_link[(nl1, nl2)] = len(new_links)
                    pos_link[(nl2, nl1)] = len(new_links)
                    new_links.append((nl1, nl2))
                    new_link_presence.append([lt0, lt1])
                    # print("\tNew link :",(nl1,nl2),"Time :",[lt0,lt1])
                else:

                    new_link_presence[pos_link[(nl1, nl2)]] += [lt0, lt1]
                    # print("\tNew link :",(nl1,nl2),"Time :",[lt0,lt1])
        print("After extension on links !")
        print("Nb nodes old :", len(self.nodes), "Nb new nodes :", len(new_nodes))
        print("Nb links old :", len(self.links), "Nb new links :", len(new_links))

        return StreamGraph(nodes=new_nodes,
                           node_presence=new_node_presence,
                           links=new_links,
                           link_presence=new_link_presence)

    ###########################################################
    #                   Isolated Nodes                        #
    ###########################################################

    def isolated_nodes(self, node_list=None):
        """
        Get isolated temporal nodes (node with temporal degree == 0)

        :return:
        """
        E = self.augmented_ordered_links()
        isolated_nodes = []
        d = defaultdict(lambda: 0)
        last_activity = {}
        if node_list is None:
            for l in E:
                c = l[0]
                if c == 1:
                    _, t0, t1, u, v = l
                    if u not in last_activity:
                        last_activity[u] = u[0]
                    if v not in last_activity:
                        last_activity[v] = v[0]
                    if d[u] == 0 and t0 != last_activity[u]:
                        isolated_nodes.append((last_activity[u], t0, u[2]))
                    if d[v] == 0 and t0 != last_activity[v]:
                        isolated_nodes.append((last_activity[v], t0, v[2]))
                    d[u] += 1
                    d[v] += 1
                else:
                    _, t1, u, v = l
                    d[u] -= 1
                    d[v] -= 1
                    if d[u] == 0:
                        last_activity[u] = t1
                    if d[v] == 0:
                        last_activity[v] = t1
            for n in last_activity:
                if n[1] != last_activity[n]:
                    isolated_nodes.append((last_activity[n], n[1], n[2]))
            for n, np in zip(self.nodes, self.node_presence):
                for t0, t1 in zip(np[::2], np[1::2]):
                    if (t0, t1, n) not in d:
                        isolated_nodes.append((t0, t1, n))
        else:
            for l in E:
                c = l[0]
                if c == 1:
                    _, t0, t1, u, v = l
                    if u[2] in node_list:
                        if u not in last_activity:
                            last_activity[u] = u[0]
                        if d[u] == 0 and t0 != last_activity[u]:
                            isolated_nodes.append((last_activity[u], t0, u[2]))
                        d[u] += 1

                    if v[2] in node_list:
                        if v not in last_activity:
                            last_activity[v] = v[0]
                        if d[v] == 0 and t0 != last_activity[v]:
                            isolated_nodes.append((last_activity[v], t0, v[2]))
                        d[v] += 1
                else:
                    _, t1, u, v = l
                    if u[2] in node_list:
                        d[u] -= 1
                        if d[u] == 0:
                            last_activity[u] = t1
                    if v[2] in node_list:
                        d[v] -= 1
                        if d[v] == 0:
                            last_activity[v] = t1
            for n in last_activity:
                if n[1] != last_activity[n]:
                    isolated_nodes.append((last_activity[n], n[1], n[2]))

            for n in node_list:
                np = self.node_presence[n]
                for t0, t1 in zip(np[::2], np[1::2]):
                    if (t0, t1, n) not in d:
                        isolated_nodes.append((t0, t1, n))
        return isolated_nodes

    ###################################
    #            MetaWalks            #
    ###################################

    def metapaths(self,x):
        """the function returns all metapaths in a link stream
        and is based on bellman-ford algorithm

        :param x: x the node from which we want the metawalks
        :return: a list of the metapaths in the link stream 
        """
        res = [set() for i in range(len(self.nodes))]

        #the 2 loops inside the first are to iterate over each temporal link 
        for k in self.nodes:
            for i in range(0,len(self.links)):
                a,b = self.links[i]
                for j in range(0,len(self.link_presence[i]),2):
                    t1,t2 = self.link_presence[i][j:j+2]
                    #print("lien actuel",a,b,t1,t2)
                    if a == x:
                        res[b].add(mw.Metawalk([(t1,t2)],[a,b]))
                    else:
                        if len(res[a]) != 0:
                            for e in res[a]:
                                start,end = e.time_intervals[-1]
                                #print("comp", t1,t2,start,end,e)
                                # check if the path can be extended
                                if start <= t1:
                                    #we should clone the metawalk and add something to it
                                    m = e.clone()
                                    m.time_intervals.append((t1,t2))
                                    m.nodes.append(b)
                                    res[b].add(m)
        return res

    def actual_degree(self,p):
        x = p.degree()
        for i in range(0,x+1):
            if p.coef[x-i] != 0:
                return x-i
        # polynomial is null
        return -1


    def betweenness_temporal_node(self,t,v):
        """il faut arranger plein de trucs, genre couper les chemins dès le début mais bon """
        self.add_point(t)
        new = self.fragmented_stream_graph()
        all_fastest = [ [ set() for j in range(len(new.nodes))] for i in range(len(new.nodes))]
        fastest_passing_through = [ [set() for j in range(len(new.nodes))] for i in range(len(new.nodes))]
        for i in range(0,len(new.nodes)):
            l = new.fastest_paths_from_vertex(new.nodes[i])
            #a changer et couper les chemins en amont
            for el in l:
                for e in el.values():
                    for ee in e:
                        ee = ee.fastest_meta_walk()
                        all_fastest[i][ee.last_node()].add(ee)
                        if ee.passes_through(t,v):
                            fastest_passing_through[i][ee.last_node()].add(ee)
        return self.matthieu_clem_betweenness(all_fastest,fastest_passing_through)

    def get_volume_betweenness(self,all_fast,passing):
        return [   [   (self.volume_metapaths(list(passing[i][j])),self.volume_metapaths(list(all_fast[i][j])))  for j in range(0,len(self.nodes))]  for i in range(0,len(self.nodes))]


    def matthieu_clem_betweenness(self,all_fast,passing):
        #not including contribution..
        print("all_fastest",all_fast)
        print("passing_thruough",passing)
        res = self.get_volume_betweenness(all_fast,passing)
        print("res",res)
        return sum(sum( self.value_betweenness_at(res,i,j)  for j in range(0,len(res[i])) )   for i in range(0,len(res)))

    def betweenneess_interval(self,interval_size):
        m = max(self.times)
        for i in range(1,m//interval_size):
            self.add_point(i*(m//interval_size))
        new = self.fragmented_stream_graph()
        all_fastest = [ [ set() for j in range(len(new.nodes))] for i in range(len(new.nodes))]
        fastest_passing_through = [ [set() for j in range(len(new.nodes))] for i in range(len(new.nodes))]
        for i in range(0,len(new.nodes)):
            l = new.fastest_paths_from_vertex(new.nodes[i])
            #a changer et couper les chemins en amont
            for el in l:
                for e in el.values():
                    for ee in e:
                        ee = ee.fastest_meta_walk()
                        all_fastest[i][ee.last_node()].add(ee)
                        if ee.passes_through(t,v):
                            fastest_passing_through[i][ee.last_node()].add(ee)
        return self.matthieu_clem_betweenness(all_fastest,fastest_passing_through)



    def value_betweenness_at(self,res,i,j):
        x,y = res[i][j]
        p = self.actual_degree(x)
        q = self.actual_degree(y)
        if p == -1:
            return 0
        if p == q:
            return x.coef[p]/y.coef[q]
        else:
            return 0



    def volume_metapaths(self,l):
        if l == []:
            return nppol.Polynomial([0])
        return sum(e.volume() for e in l)

    def portion_sorted_list(self,l,a,b):
        b1 = False
        b2 = True
        #we should not have any element
        if a == b:
            return (0,0)
        t1 = 0
        t2 = len(l)
        for i in range(0,len(l)):
            if (l[i] > a) and (not b1):
                t1 = i
                b1 = True
            if (l[i] >= b) and b2:
                t2 = i 
                b2 = False
        return (t1,t2)

    def __deepcopy__(self):
        """
        make a deepcopy of a stream graph
        """
        id = self.id
        times = copy.deepcopy(self.times)
        nodes = copy.deepcopy(self.nodes)
        #node_to_label = dict()
        # for e in self.node_to_label.keys():
        #     node_to_label[e] = self.node_to_label[e]
        node_to_label = self.node_to_label.copy()
        #print(type(self.node_to_id))
        if self.node_to_id != None:
            node_to_id = self.node_to_id.copy()
        else:
            node_to_id = None
        nodes_presence = copy.deepcopy(self.node_presence)
        links = copy.deepcopy(self.links)
        link_presence = copy.deepcopy(self.link_presence)
        weights = copy.deepcopy(self.weights)
        trips = copy.deepcopy(self.trips)
        points = self.points[:]
        return StreamGraph(id,times,nodes,node_to_label,node_to_id,nodes_presence,links,link_presence,weights,trips,points,self.alpha,self.omega)

    def duplicate_elem_in_list(self, l):
        res = l + l
        res.sort()
        return res

    def fragmented_stream_graph(self):
        """Fragments a link stream so that links do not interlap, they either happens at the exact same interval or between different intervals, it creates a new link stream"""
        s = self.__deepcopy__()
        l = list(self.event_times())
        l.sort()
        for i in range(0,len(self.links)):
            a,b = self.links[i]
            decalage = 0
            for j in range(0,len(self.link_presence[i]),2):
                t1,t2 = self.link_presence[i][j:j+2]
                #print("t1,t2",a,b,(t1,t2))
                por_1, por_2 = self.portion_sorted_list(l,t1,t2)
                portion = l[por_1:por_2]
                #print("portion",por_1,por_2,l[por_1:por_2])
                dup = self.duplicate_elem_in_list(portion)
                dup_interval = [0 if k%2 == 0 else 1 for k in range(0,len(dup))]
                s.link_presence[i] = s.link_presence[i][0:j+1+decalage] + dup + s.link_presence[i][j+1+decalage:len(s.link_presence[i])]
                decalage += len(dup)

        return s

    def add_point(self,t):
        """
        Add a point to a StreamGraph
        :param t: the time to be added in the event times
        """
        if t not in self.points:
            self.points.append(t)

    def create_dictionary_all_metapaths(self, mp):
        res = [dict() for j in range(len(self.nodes))] 
        for i in range(0,len(mp)):
            for m in mp[i]:
                s,e = (m.last_departure(),m.first_arrival())
                if (s,e) not in res[i]:
                    res[i][(s,e)] = set()
                    res[i][(s,e)].add(m)
                else:
                    res[i][(s,e)].add(m)

        return res

    def update_metapaths(self, final_paths,  b, t1, t2, m):
        #normally there should be no keys (i.e intervals) in the paths included in others
        intervals = [e for e in list(final_paths[b].keys()) ]
#        print("update_metapaths",b,t1,t2)
        boo = True
        if intervals == []:
            final_paths[b][(t1,t2)] = set()
            final_paths[b][(t1,t2)].add(m)
        else:

            if t2 <= t1:
                # example (3,1) for t1 t2, in this case t1,t2 is a metaedge and nothing happens in the fls in this time, only in boundaries things can happen. if it is an instantenous metapaths it would be included in this one, else it is not an instantenous metapaths and if it starts during t1,t2 it has to be omitted.
                for e in intervals:
                    a,c = e
                    if a >= t2 and a <= t1 and a <= c: #remove instantenous paths included in this one and non instantenous ones.
 #                       print("******** deleting *******",e)
                        del final_paths[b][e]
                    elif c >= t2 and c <= t1 and a <= c:
                        #on peut en avoir plusieur, ne pas sortir ici
                        del final_paths[b][e]
                if (t1,t2) in final_paths[b]:
                    l = list(final_paths[b][(t1,t2)])
                    if len(m.time_intervals) < len(l[0].time_intervals):
                        final_paths[b][e] = set()
                        final_paths[b][e].add(m)
                    elif len(m.time_intervals) == len(l[0].time_intervals):
                        final_paths[b][(t1,t2)].add(m)
                    return

            else: #non instantenous metawalk
                for e in intervals:
                    a,c = e
                    if (a >= t1 and c < t2) or (a > t1 and c <= t2):
                        boo = False
                        return boo
                    if (t1 >= a and t2 < c) or (t1 > a and t2 <= c):
                        #on peut en avoir plusieur, ne pas sortir ici
                        del final_paths[b][e]
                    if (t1 == a and t2 == c):
                        l = list(final_paths[b][e])
                        if len(m.time_intervals) < len(l[0].time_intervals):
                            final_paths[b][e] = set()
                            final_paths[b][e].add(m)
                        elif len(m.time_intervals) == len(l[0].time_intervals):
                            final_paths[b][e].add(m)
                        return
            final_paths[b][(t1,t2)] = set()
            final_paths[b][(t1,t2)].add(m)


    def is_latency(self, res,  b, t1, t2):
        # the pair is instantenous
        if t2 <= t1:
            return True
        #check for inclusion
        existing_keys = [e for e in list(res[b].keys()) if (((e[0] > t1) and (e[1] <= t2)) or ((e[0] >= t1) and (e[1] < t2))) and (not (e[0] == t1 and e[1] == t2)) ]
        return existing_keys == []

    def filter_fastest_metapaths_slow(self, m):
        res = self.create_dictionary_all_metapaths(m)
        res2 = self.create_dictionary_all_metapaths(m)
        for b in range(0,len(self.nodes)):
            for (t1,t2) in res[b]:
                if t1 < t2:
                    if not self.is_latency(res,b,t1,t2):
                       # print(b,t1,t2)
                    # there exists a faster path
                        if (t1,t2) in res2[b]:
                            del res2[b][(t1,t2)]
        return res2

    def filter_metapaths(self,x,b,t1,t2,res):
        #print("filter",x,b,t1,t2)
        # for el in res:
        #     for e in el.items():
        #         for ee in e:
        #             print(ee)
        # print("fin dict")

        # i think we can do better, by filtering on the fly not after
        # check if latency

            existing_keys = [e for e in list(res[b].keys()) if (((e[0] > t1) and (e[1] <= t2)) or ((e[0] >= t1) and (e[1] < t2))) and (not (e[0] == t1 and e[1] == t2)) ]


            if existing_keys != [] and t1 < t2:
                #not latency
                del res[b][(t1,t2)]
            else:
                #it is a latency and we need to delete other pairs
                existing_keys = [e for e in list(res[b].keys()) if ((e[0] == t1) and (e[1] > t2)) or ((e[0] <  t1) and (e[1] == t2)) and (e[0] != e[1]) ]
               # print("existing",existing_keys)
                for e in existing_keys:
                    # its a latency and we remove slowlier paths
                    del res[b][e]
#        return res



    def check_edge(self,x,a,b,t1,t2,res, res2):
        #print("check_edge",a,b,t1,t2)
        if a == x:
                #we should check there is later departure with same arrival but only one edge used in this case so it seems ok
            if (t2,t1) not in res[b]:
                res[b][(t2,t1)] = set()
                res2[b][(t2,t1)] = set()
            res[b][(t2,t1)].add(mw.Metawalk([(t1,t2)],[a,b]))
            res2[b][(t2,t1)].add(mw.Metawalk([(t1,t2)],[a,b]))
        else:
            if len(res2[a]) != 0:
                ll = f.reduce(lambda a, b: a+b, list(map(list,list(res2[a].values()))))
                for e in ll:
                    start,end = e.time_intervals[-1]
                    # check if the path can be extended
                    if start <= t1:
                        #we should clone the metawalk and add something to it
                        m = e.clone()
                        m.time_intervals.append((t1,t2))
                        m.nodes.append(b)
                        last_depar = m.last_departure()
                        if (last_depar,t1) not in res[b]:
                            res[b][(last_depar,t1)] = set()
                            res[b][(last_depar,t1)].add(m)
                        else:
                            res[b][(last_depar,t1)].add(m)

                        if (last_depar,t1) not in res2[b]:
                            res2[b][(last_depar,t1)] = set()
                            res2[b][(last_depar,t1)].add(m)
                        else:
                            res2[b][(last_depar,t1)].add(m)
                        #print("edge",m)
                        self.filter_metapaths(x,b,last_depar,t1,res)

    def sf_paths_check(self,x,a,b,t1,t2,res, res2):
        if a == x:
                #we should check there is later departure with same arrival but only one edge used in this case so it seems ok
            if (t2,t1) not in res[b]:
                res[b][(t2,t1)] = set()
                res2[b][(t2,t1)] = set()
            res[b][(t2,t1)].add(mw.Metawalk([(t1,t2)],[a,b]))
            res2[b][(t2,t1)].add(mw.Metawalk([(t1,t2)],[a,b]))

    def adapt_metawalk(self,start,end,t1,t2,e,b):
            # check if the path can be extended
        if start <= t1 and end <= t2: #the extension can be added with no problems
                                #we should clone the metawalk and add something to it
            m = e.clone()
            m.time_intervals.append((t1,t2))
            m.nodes.append(b)
        elif start <= t1:
            m = e.clone()
            m.time_intervals.append((t1,t2))
            m.nodes.append(b)
            m.update_following_last(1)
        elif end <= t2:
            m = e.clone()
            m.time_intervals.append((t1,t2))
            m.nodes.append(b)
            m.update_following_last(0)
        else:
            m = None
        return m

    def is_shortest_path(self,last_depar,t1,final_paths,m):
        if (last_depar,t1) in final_paths[b]:
            r = list(final_paths[b])[0]
            if len(r.time_intervals) < len(m.time_intervals):
                return False
        return True

    def extend_paths(self,x,a,b,t1,t2,final_paths, tmp_paths):
        #print("extend_paths",a,b,t1,t2)
        #print("keys")
        #for e in final_paths:
        #    print(e.keys())
        #print("final",final_paths)
        if a == x:
            if (t2,t1) not in final_paths[b]:
                final_paths[b][(t2,t1)] = set()
                tmp_paths[b][(t2,t1)] = set()
            final_paths[b][(t2,t1)].add(mw.Metawalk([(t1,t2)],[a,b]))
            tmp_paths[b][(t2,t1)].add(mw.Metawalk([(t1,t2)],[a,b]))
        else:
            if len(tmp_paths[a]) != 0:
                ll = f.reduce(lambda a, b: a+b, list(map(list,list(tmp_paths[a].values()))))
                for e in ll:
                    start,end = e.time_intervals[-1]
                    boo = True
                    m = self.adapt_metawalk(start,end,t1,t2,e,b)
                    #print("res adapt",e,m)
                    if m == None :
                        boo = False
                    if boo == True:
                        last_depar = m.last_departure()
                        if ( b in e.nodes):
                            return
                        #in all cases the path is ok, so it is added to tmp paths
                        if (last_depar,t1) not in tmp_paths[b]:
                            tmp_paths[b][(last_depar,t1)] = set()
                            tmp_paths[b][(last_depar,t1)].add(m)
                        else:
                            tmp_paths[b][(last_depar,t1)].add(m)
                        # t1 is first arrival
                        self.update_metapaths(final_paths,  b, last_depar, t1, m)

    def arrival(self, e, t1, t2):
        second_arrival = -1
        edge_taken2 = -1
        if t2 >= e:

            if t2 == e:
                edge_taken1 = (t1,t2)
                first_arrival = e
            else:
                if t1 != t2:
                    second_arrival = t2
                first_arrival = t1
                edge_taken1 = (t1,t1)
                edge_taken2 = (t1,t2)
            res = (first_arrival, second_arrival, edge_taken1,edge_taken2)
        else:
            res = (-1,-1,-1,-1)
        return res

    def compute_c(self, last_depar, arrival, leng):
        if arrival - last_depar < 0:
            return (0.0,leng)
        else:
            return (arrival - last_depar,leng)

    def closest_arrival(self,arrival,arrivals_b):
        if len(arrivals_b) == 0:
            return -1
        elif len(arrivals_b) == 1:
            if arrivals_b[0] <= arrival:
                return arrivals_b[0]
            else:
                return -1
        else:
            if arrivals_b[-1] <= arrival:
                return arrivals_b[-1]
            for i in range(len(arrivals_b)-1,1,-1):
                if (arrivals_b[i-1] <= arrival) and (arrivals_b[i] >= arrival):
                    return arrivals_b[i-1]

            else:
                return -1

    def relax_resting_aux(self, b, last_depar, arrival, e, cur_best, pre):
        if arrival in cur_best[b]:
            cnew = self.compute_c(last_depar, arrival, cur_best[b][e][1])
            cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
            #if c < ((arrival - cur_best[b][arrival][0]), cur_best[b][arrival][1]):
            if cnew < cold:
                #pre[b][arrival] = set()
                pre[b][arrival] = dict()
                cur_best[b][arrival] = (last_depar,cur_best[b][e][1])
        else:
            #pre[b][arrival] = set()
            pre[b][arrival] = dict()
            cur_best[b][arrival] =  (last_depar,cur_best[b][e][1])
            #pre[b][arrival].add((a,e,edge_taken1))
            #pre[b][arrival][(a,e)] = edge_taken

    def relax_resting_paths(self, b, t1, t2, pre, cur_best, events, events_rev):
        #arrivals is a dictionary of integers that represent last_arrival, the value of each element in the dict is a couple (last_departure, length of metawalk)
        arrivals_b = [e for e in events if (e in pre[b]) and (pre[b][e] != {}) ]
        #arrivals_b.sort()
        #l = events
        # D = dict()
        # for i in range(0,len(events)):
        #     D[l[i]] = i
        #print("relaxing_paths", b, t1, t2)
        #print("resting",arrivals_b)
        close_arrival = self.closest_arrival(t2, arrivals_b)
        #print("close_arrival",close_arrival, t2)
        if close_arrival != -1:
            last_depar = cur_best[b][close_arrival][0]
            for i in range(events_rev[close_arrival],len(events)):
                self.relax_resting_aux(b, last_depar, events[i], close_arrival, cur_best, pre)

            #e = pre[b][close_arrival][1]
            #aa = pre[b][close_arrival][0]
            # if t1 != t2:
            #     first_arrival, second_arrival= (t1, t2)
            # else:
            #     first_arrival, second_arrival= (t1, -1)
            #print("relaxing_paths")
            #print(aa,b,t1,t2)
            #print("cur_bestb",cur_best[b])
            #print("preb",pre[b])
            #print("arrivals",arrivals_b)
            #print("first,second",first_arrival, second_arrival)
            #print("close_arrival",close_arrival, "last_depar", last_depar)

            #if first_arrival in cur_best[b]:
            #self.relax_resting_aux(b, last_depar, first_arrival, close_arrival, cur_best, pre)
            #     cnew = self.compute_c(last_depar, first_arrival, cur_best[b][close_arrival][1])
            #     cold = self.compute_c(cur_best[b][first_arrival][0], first_arrival, cur_best[b][first_arrival][1])
            #     #if c < ((first_arrival - cur_best[b][first_arrival][0]), cur_best[b][first_arrival][1]):
            #     if cnew < cold:
            #         #pre[b][first_arrival] = set()
            #         pre[b][first_arrival] = dict()
            #         cur_best[b][first_arrival] = (last_depar,cur_best[b][close_arrival][1])
            #         #pointer[(b,first_arrival)] = (b,close_arrival) 
            #         #cold = self.compute_c(cur_best[b][first_arrival][0], first_arrival, cur_best[b][first_arrival][1])
            #         # print("comp", cnew,((first_arrival - cur_best[b][first_arrival][0]) , cur_best[b][first_arrival][1]))
            #     # if cnew == cold:
            #     #     if (aa,e) not in pre[b][first_arrival]:
            #     #         pre[b][first_arrival][(aa,e)] = pre[b][close_arrival][edge_last]
            #     #         bool = True
            #     #     else:
            #     #         e1,e2 = edge_last
            #     #         x1,x2 = pre[b][first_arrival][(aa,e)]
            #     #         if (e1 == x1 and e2 > x2) or (e2 == x2 and x1 < e1):
            #     #             pre[b][first_arrival][(aa,e)] = edge_last
            #     #             bool = True

            # else:
            #     pre[b][first_arrival] = dict()
            #     cur_best[b][first_arrival] =  (last_depar,cur_best[b][close_arrival][1] )
            #     #pointer[(b,first_arrival)] = (b,close_arrival) 
            #     #pre[b][first_arrival].add((a,e,edge_taken1))
            #     # pre[b][first_arrival][(aa,e)] = edge_last
            #     # bool = True


            #if second_arrival != -1:
            #    self.relax_resting_aux(b, last_depar, second_arrival, close_arrival, cur_best, pre)
                # if second_arrival in cur_best[b]:
                #     c2new = self.compute_c(last_depar, second_arrival, cur_best[b][close_arrival][1])
                #     c2old = self.compute_c(cur_best[b][second_arrival][0], second_arrival, cur_best[b][second_arrival][1])
                #     #c2 = ( (second_arrival - last_depar) , cur_best[a][e][1] + 1)
                #     # if (second_arrival - last_depar) <= 0:
                #     #     c2 = (0.0,cur_best[a][e][1] + 1)
                #     # else:
                #     #     c2 = ( (second_arrival - last_depar) , cur_best[a][e][1] + 1)

                #     if c2new < c2old:
                #         pre[b][second_arrival] = dict()
                #         cur_best[b][second_arrival] = (last_depar,cur_best[b][close_arrival][1])
                #         #pointer[(b,second_arrival)] = (b,close_arrival) 
                #         # print("comp", c2new,((second_arrival - cur_best[b][second_arrival][0]) , cur_best[b][second_arrival][1]))
                #         #c2old = self.compute_c(cur_best[b][second_arrival][0], second_arrival, cur_best[b][second_arrival][1])

                #         # if c2new == c2old:
                #         #     if (aa,e) not in pre[b][second_arrival]:
                #         #         if bool:
                #         #             pre[b][second_arrival][(aa,e)] = edge_last
                #         #         else:
                #         #             pre[b][second_arrival][(aa,e)] = edge_last
                #         #     else:
                #         #         e1,e2 = edge_last
                #         #         x1,x2 = pre[b][second_arrival][(aa,e)]
                #         #         if (e1 == x1 and e2 > x2) or (e2 == x2 and x1 < e1):
                #         #             if bool:
                #         #                 pre[b][second_arrival][(aa,e)] = edge_last
                #         #             else:
                #         #                 pre[b][second_arrival][(aa,e)] = edge_last
                # else:
                #     pre[b][second_arrival] = dict()
                #     cur_best[b][second_arrival] =  (last_depar,cur_best[b][close_arrival][1])
                #     #pointer[(b,second_arrival)] = (b,close_arrival) 
                #     # if bool:
                #     #     pre[b][second_arrival][(aa,e)] = edge_last
                #     # else:
                #     #     pre[b][second_arrival][(aa,e)] = edge_last
        return

    def relax_paths_aux(self, a, b, last_depar, arrival, e, edge_taken, cur_best, pre):
        if arrival in cur_best[b]:
            cnew = self.compute_c(last_depar, arrival, cur_best[a][e][1] + 1)
            cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
            #if c < ((arrival - cur_best[b][arrival][0]), cur_best[b][arrival][1]):
            if cnew < cold:
                #pre[b][arrival] = set()
                pre[b][arrival] = dict()
                cur_best[b][arrival] = (last_depar,cur_best[a][e][1] + 1)
                cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
            if cnew == cold:
                #pre[b][arrival].add((a,e,edge_taken))
                if (a,e) not in pre[b][arrival]:
                    pre[b][arrival][(a,e)] = edge_taken
                else:
                    e1,e2 = edge_taken
                    x1,x2 = pre[b][arrival][(a,e)]
                    if (e1 == x1 and e2 > x2) or (e2 == x2 and e1 < x1):
                        pre[b][arrival][(a,e)] = edge_taken
        else:
            #pre[b][arrival] = set()
            pre[b][arrival] = dict()
            cur_best[b][arrival] =  (last_depar,cur_best[a][e][1] + 1)
            #pre[b][arrival].add((a,e,edge_taken1))
            pre[b][arrival][(a,e)] = edge_taken


    def relax_paper(self, a, b, t1, t2, pre, cur_best, events):
        #arrivals is a dictionary of integers that represent last_arrival, the value of each element in the dict is a couple (last_departure, length of metawalk)
        arrivals = [e for e in events if (e in pre[a]) and (pre[a][e] != {})]
        # print(a,b,t1,t2)
        # print("cur_besta",cur_best[a])
        # print("cur_bestb",cur_best[b])
        # print("prea",pre[a])
        # print("preb",pre[b])
        for i in range(len(arrivals)):
            e = arrivals[i]
            #print("arrivals",arrivals)
            last_depar = cur_best[a][e][0]
            first_arrival, second_arrival, edge_taken1, edge_taken2 = self.arrival(e, t1, t2)
            #print("edge",first_arrival,second_arrival,edge_taken1,edge_taken2)
            # if t2 >= e: # the new metawalk is ok
            #     if t2 == e : #3 cases possible on intervals positions
            #         if cur_best[a][e][0] > t2:
            #             last_depar = t2
            #         else:
            #             last_depar = cur_best[a][e][0]
            #         edge_taken = (t2,t2)
            #         first_arrival = e
            #     if t1 == e and t2 != e : #3 cases possible on intervals positions
            #         first_arrival = e
            #         second_arrival = t2
            #         last_depar = cur_best[a][e][0]
            #         edge_taken = (t1,t2)

            #         #first_arrival = e
            #     elif t1 > e:
            #         if t1 != t2:
            #             second_arrival = t2
            #         edge_taken = (t1,t2)
            #         first_arrival = t1
            #         last_depar = cur_best[a][e][0]
            #         #print("first_arrival",first_arrival, "c",c,cur_best[b],maxi)
            if (first_arrival, second_arrival, edge_taken1, edge_taken2) != (-1,-1,-1,-1):
                self.relax_paths_aux(a, b, last_depar, first_arrival, e, edge_taken1, cur_best, pre)
                #c = ( (first_arrival - last_depar) , cur_best[a][e][1] + 1)
                # if (first_arrival - last_depar) <= 0:
                #     c = (0.0, cur_best[a][e][1] + 1)
                # else:
                #     c = ( (first_arrival - last_depar) , cur_best[a][e][1] + 1)
                # if first_arrival in cur_best[b]:
                #     cnew = self.compute_c(last_depar, first_arrival, cur_best[a][e][1] + 1)
                #     cold = self.compute_c(cur_best[b][first_arrival][0], first_arrival, cur_best[b][first_arrival][1])
                #     #if c < ((first_arrival - cur_best[b][first_arrival][0]), cur_best[b][first_arrival][1]):
                #     if cnew < cold:
                #         #pre[b][first_arrival] = set()
                #         pre[b][first_arrival] = dict()
                #         cur_best[b][first_arrival] = (last_depar,cur_best[a][e][1] + 1)
                #         #pointer[(b,first_arrival)] = (b,first_arrival) 
                #         cold = self.compute_c(cur_best[b][first_arrival][0], first_arrival, cur_best[b][first_arrival][1])
                #         #print("comp", cnew,((first_arrival - cur_best[b][first_arrival][0]) , cur_best[b][first_arrival][1]))
                #     # if (first_arrival - cur_best[b][first_arrival][0]) < 0:
                #     #     x = 0
                #     # else:
                #     #     x = (first_arrival - cur_best[b][first_arrival][0])
                #     # if c == (x , cur_best[b][first_arrival][1]):
                #     #if c == ((first_arrival - cur_best[b][first_arrival][0]), cur_best[b][first_arrival][1]):
                #     if cnew == cold:
                #         #pre[b][first_arrival].add((a,e,edge_taken))
                #         if (a,e) not in pre[b][first_arrival]:
                #             pre[b][first_arrival][(a,e)] = edge_taken1
                #             bool = True
                #         else:
                #             e1,e2 = edge_taken1
                #             x1,x2 = pre[b][first_arrival][(a,e)]
                #             if (e1 == x1 and e2 > x2) or (e2 == x2 and e1 < x1):
                #                 pre[b][first_arrival][(a,e)] = edge_taken1
                #                 bool = True

                # else:

                #     #pre[b][first_arrival] = set()
                #     pre[b][first_arrival] = dict()
                #     cur_best[b][first_arrival] =  (last_depar,cur_best[a][e][1] + 1)
                #     #pre[b][first_arrival].add((a,e,edge_taken1))
                #     pre[b][first_arrival][(a,e)] = edge_taken1
                #     #pointer[(b,first_arrival)] = (b,first_arrival) 
                #     bool = True


                if second_arrival != -1:
                    self.relax_paths_aux(a, b, last_depar, second_arrival, e, edge_taken2, cur_best, pre)
                    # if second_arrival in cur_best[b]:
                    #     c2new = self.compute_c(last_depar, second_arrival, cur_best[a][e][1] + 1)
                    #     c2old = self.compute_c(cur_best[b][second_arrival][0], second_arrival, cur_best[b][second_arrival][1])
                    #     #c2 = ( (second_arrival - last_depar) , cur_best[a][e][1] + 1)
                    #     # if (second_arrival - last_depar) <= 0:
                    #     #     c2 = (0.0,cur_best[a][e][1] + 1)
                    #     # else:
                    #     #     c2 = ( (second_arrival - last_depar) , cur_best[a][e][1] + 1)

                    #     if c2new < c2old:
                    #         pre[b][second_arrival] = dict()
                    #         cur_best[b][second_arrival] = (last_depar,cur_best[a][e][1] + 1)
                    #         #pointer[(b,second_arrival)] = (b,second_arrival) 
                    #         #print("comp", c2new,((second_arrival - cur_best[b][second_arrival][0]) , cur_best[b][second_arrival][1]))
                    #         c2old = self.compute_c(cur_best[b][second_arrival][0], second_arrival, cur_best[b][second_arrival][1])
                    #     # if (second_arrival - cur_best[b][second_arrival][0]) < 0:
                    #     #     x = 0
                    #     # else:
                    #     #     x = (second_arrival - cur_best[b][second_arrival][0])
                    #     # if c2 == (x , cur_best[b][second_arrival][1]):
                    #     if c2new == c2old:
                    #         #pre[b][second_arrival].add((a,e,edge_taken))
                    #         if (a,e) not in pre[b][second_arrival]:
                    #             if bool:
                    #                 pre[b][second_arrival][(b,e)] = edge_taken2
                    #             else:
                    #                 pre[b][second_arrival][(a,e)] = edge_taken2
                    #         else:
                    #             e1,e2 = edge_taken2
                    #             x1,x2 = pre[b][second_arrival][(a,e)]
                    #             if (e1 == x1 and e2 > x2) or (e2 == x2 and e1 < x1):
                    #                 if bool:
                    #                     pre[b][second_arrival][(b,e)] = edge_taken2
                    #                 else:
                    #                     pre[b][second_arrival][(a,e)] = edge_taken2
                    # else:
                    #     pre[b][second_arrival] = dict()
                    #     cur_best[b][second_arrival] =  (last_depar,cur_best[a][e][1] + 1)
                    #     #pointer[(b,second_arrival)] = (b,second_arrival) 
                    #     if bool:
                    #         pre[b][second_arrival][(b,e)] = edge_taken2
                    #     else:
                    #         pre[b][second_arrival][(a,e)] = edge_taken2
        return
        # def add_curbest_initialise_pre(self, b, t, cur_best, pre):
    #     if t not in cur_best[b]:
    #         cur_best[b][t] = (t,1)
    #     else:
    #         if (0.0,1) < (t - cur_best[b][t][0],cur_best[b][t][1]):
    #             cur_best[b][t] = (t,1)
    #             pre[b][t] = dict()
    #     return


    # def add_from_origin(self, a, b, t1, t2, pre, cur_best):
    #     self.add_curbest_initialise_pre(b, t1, cur_best, pre)
    #     self.add_curbest_initialise_pre(b, t2, cur_best, pre)
    #     if t1 not in pre[b]:
    #         pre[b][t1] = {(a,0.0):(t1,t1)}
    #     else:
    #         #no need for else case, since this can only be included
    #         if (a,0.0) not in pre[b][t1]:
    #             pre[b][t1][(a,0.0)] = (t1,t1)
    #     if t2 not in pre[b]:
    #         pre[b][t2] = {(a,0.0):(t1,t2)}
    #     else:
    #         if (a,0.0) in pre[b][t2]:
    #             e1,e2 = pre[b][t2][(a,0.0)]
    #             if (t1 == e1 and t2 > e2) or (t1 < e1 and t2 == e2):
    #                 pre[b][t2][(a,0.0)] = (t1,t2)
    #                 #pre[b][t2] = {(b,0.0):(t1,t2)}

    def count_walks_paper(self,x, events, events_rev):
        cur_best = [ {t:(-numpy.Infinity,numpy.Infinity)   for t in events} for i in range(len(self.nodes)) ]
        pre = [{t:{}   for t in events} for i in range(len(self.nodes))]
        for e in cur_best[x]:
            cur_best[x][e] = (e,0)
            pre[x][e]={(0,0):(-1,-1)}
        #the 2 loops inside the first are to iterate over each temporal link 
        for k in self.nodes:
            for i in range(0,len(self.links)):
                a,b = self.links[i]
                for j in range(0,len(self.link_presence[i]),2):
                    t1,t2 = self.link_presence[i][j:j+2]
                    self.relax_paper(a,b,t1,t2,pre,cur_best, events)
                    self.relax_resting_paths(b,t1,t2,pre,cur_best, events, events_rev)
                    self.relax_paper(b,a,t1,t2,pre,cur_best, events)
                    self.relax_resting_paths(a,t1,t2,pre,cur_best, events, events_rev)
                        #relax_paper(self,x,b,a,t1,t2,pre,cur_best,maxi)

        return (pre, cur_best)


    def relax_resting_aux_dij(self, b, t, arrival, cur_best, pre, Q, Q_nod):
        #if arrival in P[b]:
        #    return
        cnew = self.compute_c(cur_best[b][t][0], arrival, cur_best[b][t][1])
        cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
        if cnew < cold:
            #pre[b][arrival] = set()
            pre[b][arrival] = dict()
            cur_best[b][arrival] = (cur_best[b][t][0],cur_best[b][t][1])
            if (b,arrival) in Q_nod:
                Q.decrease_key(Q_nod[b,arrival], (cnew,(b,arrival)))
            else:
                Q_nod[b,arrival] = Q.insert( (cnew,(b,arrival) ) )

    def relax_resting_paths_dij(self, b, t, t1, t2, pre, cur_best, events, events_rev, Q, Q_nod):
        for i in [events_rev[t1],events_rev[t2]]:
            if events[i] >= t:
                self.relax_resting_aux_dij(b, t, events[i], cur_best, pre, Q, Q_nod)

        return

    def relax_resting_paths_dij2(self, b, t, tp, pre, cur_best, events, events_rev, Q, Q_nod):
        self.relax_resting_aux_dij(b, t, tp, cur_best, pre, Q, Q_nod)

        return

    def relax_paths_aux_dij(self, a, b, last_depar, arrival, e, edge_taken, cur_best, pre, Q, Q_nod):
        print("a",a,"e",e,"b",b,"arrival",arrival)
        #if arrival in P[b]:
        #    return
        cnew = self.compute_c(last_depar, arrival, cur_best[a][e][1] + 1)
        cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
        print("cnew",cnew,"cold",cold)
        #if c < ((arrival - cur_best[b][arrival][0]), cur_best[b][arrival][1]):
        if cnew < cold:
            #pre[b][arrival] = set()
            pre[b][arrival] = dict()
            cur_best[b][arrival] = (last_depar,cur_best[a][e][1] + 1)
            cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
            #print("Q_nod[b,arrival]",Q_nod[b,arrival])
            if (b,arrival) in Q_nod:
                Q.decrease_key(Q_nod[b,arrival], (cnew,(b,arrival)))
            else:
                Q_nod[b,arrival] = Q.insert( (cnew,(b,arrival) ) )
            #if last_depar not in distance[b]:
            #    distance[b][last_depar] = dict()
            #distance[b][last_depar][cnew[1]] = arrival

            #Q.decrease_key(Q_nod[b,arrival], (cnew, (b,arrival)) )
        if cnew == cold:
            #pre[b][arrival].add((a,e,edge_taken))
            if (a,e) not in pre[b][arrival]:
                pre[b][arrival][(a,e)] = edge_taken
            else:
                e1,e2 = edge_taken
                x1,x2 = pre[b][arrival][(a,e)]
                if (e1 == x1 and e2 > x2) or (e2 == x2 and e1 < x1):
                    pre[b][arrival][(a,e)] = edge_taken

    def relax_paths_aux_dij2(self, a, b, last_depar, arrival, e, edge_taken, cur_best, pre, Q, Q_nod):
        print("a",a,"e",e,"b",b,"arrival",arrival)
        #if arrival in P[b]:
        #    return
        cnew = self.compute_c(last_depar, arrival, cur_best[a][e][1] + 1)
        cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
        print("cnew",cnew,"cold",cold)
        #if c < ((arrival - cur_best[b][arrival][0]), cur_best[b][arrival][1]):
        if cnew < cold:
            #pre[b][arrival] = set()
            pre[b][arrival] = dict()
            cur_best[b][arrival] = (last_depar,cur_best[a][e][1] + 1)
            cold = self.compute_c(cur_best[b][arrival][0], arrival, cur_best[b][arrival][1])
            #print("Q_nod[b,arrival]",Q_nod[b,arrival])
            if (b,arrival) in Q_nod:
                Q.decrease_key(Q_nod[b,arrival], (cnew,(b,arrival)))
            else:
                Q_nod[b,arrival] = Q.insert( (cnew,(b,arrival) ) )
            #if last_depar not in distance[b]:
            #    distance[b][last_depar] = dict()
            #distance[b][last_depar][cnew[1]] = arrival

            #Q.decrease_key(Q_nod[b,arrival], (cnew, (b,arrival)) )
        if cnew == cold:
            #pre[b][arrival].add((a,e,edge_taken))
            #if (a,e) not in pre[b][arrival]:
            pre[b][arrival][(a,e)] = edge_taken

    def relax_paper_dij(self, a, b, t, t1, t2, pre, cur_best, events, Q, Q_nod):
        #arrivals = [e for e in events if (e in pre[a]) and (pre[a][e] != {})]
        arrivals = []
        if pre[a][t] != {}:
            arrivals = [t ]
        for i in range(len(arrivals)):
            e = arrivals[i]
            #print("arrivals",arrivals)
            last_depar = cur_best[a][e][0]
            first_arrival, second_arrival, edge_taken1, edge_taken2 = self.arrival(e, t1, t2)
            if (first_arrival, second_arrival, edge_taken1, edge_taken2) != (-1,-1,-1,-1):
                self.relax_paths_aux_dij(a, b, last_depar, first_arrival, e, edge_taken1, cur_best, pre, Q, Q_nod)

                if second_arrival != -1:
                    self.relax_paths_aux_dij(a, b, last_depar, second_arrival, e, edge_taken2, cur_best, pre, Q, Q_nod)
        return

    def relax_paper_dij2(self, a, b, t, tp, pre, cur_best, events, Q, Q_nod, edge):
        if pre[a][t] == {}:
            return
        #print("arrivals",arrivals)
        last_depar = cur_best[a][t][0]
        self.relax_paths_aux_dij2(a, b, last_depar, tp, t, edge, cur_best, pre, Q, Q_nod)
        return


    def neighbors(self):
        res =[[] for i in range(len(self.nodes))]
        for e in self.links:
            x,y = e
            res[x].append(y)
            res[y].append(x)
        return res

    def neighbors_direct(self):
        res ={ i:dict() for i in self.nodes}
        res_inv = { i:dict() for i in self.nodes}
        for i in range(len(self.links)):
            x,y = self.links[i]
            if y not in res[x]:
                res[x][y] = []
            if x not in res_inv[y]:
                res_inv[y][x] = []
            #print(self.link_presence[i])
            for j in range(0,len(self.link_presence[i]),2):
                t1,t2 = self.link_presence[i][j:j+2]
                if t1 != t2:
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res[x][y].append([t1, (t1,t1)])
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res_inv[y][x].append([t1, (t1,t1)])
                    res[x][y].append([t2, (t1,t2)])
                    res_inv[y][x].append([t2, (t1,t2)])
                else:
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res[x][y].append([t1, (t1,t1)])
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res_inv[y][x].append([t1, (t1,t1)])

        return res,res_inv

    def neighbors2(self):
        res ={ i:dict() for i in self.nodes}
        for i in range(len(self.links)):
            x,y = self.links[i]
            if y not in res[x]:
                res[x][y] = []
            if x not in res[y]:
                res[y][x] = []
            #print(self.link_presence[i])
            for j in range(0,len(self.link_presence[i]),2):
                t1,t2 = self.link_presence[i][j:j+2]
                if t1 != t2:
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res[x][y].append([t1, (t1,t1)])
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res[y][x].append([t1, (t1,t1)])
                    res[x][y].append([t2, (t1,t2)])
                    res[y][x].append([t2, (t1,t2)])
                else:
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res[x][y].append([t1, (t1,t1)])
                    if (j > 0 and t1 != self.link_presence[i][j-1]) or (j==0):
                        res[y][x].append([t1, (t1,t1)])

        return res

    def link_index(self):
        d = dict()
        for i in range(0, len(self.links)):
            d[self.links[i]] = i
        return d

    def dijkstra_temporal(self, s, events, events_rev, neighbors, d):
        Q = fib.FibonacciHeap()
        cur_best = [ {t:(-numpy.Infinity,numpy.Infinity)   for t in events} for i in range(len(self.nodes)) ]
        pre = [{t:{}   for t in events} for i in range(len(self.nodes))]
        nod = dict()
        P = dict()
        distance = dict()
        for v in self.nodes:
            P[v] = set()
            distance[v] = dict()
        # for v in self.nodes:
        #     cur_best[v] = dict()
        #     pre[v] = dict()
        #     for t in events:
        #         cur_best[v][t] = (-numpy.Infinity,numpy.Infinity)
        #         pre[v][t] = {}
        #         if v == s:
        #             cur_best[v][t] = (t,0)
        #             pre[v][t]={(0,0):(-1,-1)}
        #         nod[v,t] = Q.insert( ((t - cur_best[v][t][0],cur_best[v][t][1]),(v,t) ) )
        for e in neighbors[s]:
            (xx,yy) = (s,e)
            if (s,e) not in d:
                (xx,yy) = (e,s)
            for j in range(0,len(self.link_presence[d[(xx,yy)]]),2):
            #for j in range(0,2,2):
                if self.link_presence[d[(xx,yy)]][j] != self.link_presence[d[(xx,yy)]][j+1]:
                    l = self.link_presence[d[(xx,yy)]][j:j+2]
                else:
                    l = self.link_presence[d[(xx,yy)]][j:j+1]
                for t in l:
                    cur_best[s][t] = (t,0)
                    pre[s][t] = {(0,0):(-1,-1)}
                    if (s,t) not in nod:
                        nod[s,t] = Q.insert( ((t - cur_best[s][t][0],cur_best[s][t][1]),(s,t) ) )
                        distance[s][t] = dict()
                        distance[s][t][numpy.Infinity] = t
        print(Q.total_nodes)
        while Q.total_nodes != 0:
            #mm = Q.find_min()
            #print("total_nodes",Q.total_nodes,"mm",mm.key)
            #q = Q.extract_min()
            #(x,y) = q.key
            print("nb_nodes", Q.total_nodes,"min",Q.find_min().data)
            (x,y) = Q.extract_min().data
            del nod[y]
            for jjj in self.nodes:
                print("pre", jjj,pre[jjj])
                print("cur", jjj,cur_best[jjj])
                #print("dernier", jjj,dernier_arrive[jjj])
            print(Q.total_nodes, "extracted", y,x)

            (a,t) = y
            P[a].add(t)
            (tp,dis) = x
            for b in neighbors[a]:
                (xx,yy) = (a,b)
                if (a,b) not in d:
                    (xx,yy) = (b,a)

                for j in range(0,len(self.link_presence[d[(xx,yy)]]),2):
                #for j in range(len(self.link_presence[d[(xx,yy)]])-1,-1,-2):
                    t1,t2 = self.link_presence[d[(xx,yy)]][j:j+2]
                    #t1,t2 = self.link_presence[d[(xx,yy)]][j-1:j+1]
                    #we can check for times if we'd like
                    if not (t > t2):
                        print("salut",(a,t),(b,t1,t2))
                        self.relax_resting_paths_dij(a,t,t1,t2,pre,cur_best, events, events_rev, Q, nod)
                        #self.relax_resting_paths_dij(b,t,t1,t2,pre,cur_best, events, events_rev, Q, nod, P, distance)
                        print(cur_best)
                        self.relax_paper_dij(a,b,t,t1,t2,pre,cur_best, events, Q, nod)
                        print("resting", cur_best)
                        #self.relax_paper(b,a,t1,t2,pre,cur_best, events)
                        #self.relax_resting_paths(a,t1,t2,pre,cur_best, events, events_rev)

        return (pre, cur_best)

    def dijkstra_temporal2(self, s, events, events_rev, neighbors, d):
        Q = fib.FibonacciHeap()
        cur_best = [ {t:(-numpy.Infinity,numpy.Infinity)   for t in events} for i in range(len(self.nodes)) ]
        pre = [{t:{}   for t in events} for i in range(len(self.nodes))]
        nod = dict()
        P = dict()
        for v in self.nodes:
            P[v] = set()
        for e in neighbors[s].keys():
            (xx,yy) = (s,e)
            if (s,e) not in d:
                (xx,yy) = (e,s)
            for j in range(0,len(self.link_presence[d[(xx,yy)]]),2):
            #for j in range(0,2,2):
                if self.link_presence[d[(xx,yy)]][j] != self.link_presence[d[(xx,yy)]][j+1]:
                    l = self.link_presence[d[(xx,yy)]][j:j+2]
                else:
                    l = self.link_presence[d[(xx,yy)]][j:j+1]
                for t in l:
                    cur_best[s][t] = (t,0)
                    pre[s][t] = {(0,0):(-1,-1)}
                    if (s,t) not in nod:
                        nod[s,t] = Q.insert( ((t - cur_best[s][t][0],cur_best[s][t][1]),(s,t) ) )
        print(Q.total_nodes)
        while Q.total_nodes != 0:
            print("nb_nodes", Q.total_nodes,"min",Q.find_min().data)
            (x,y) = Q.extract_min().data
            del nod[y]
            for jjj in self.nodes:
                print("pre", jjj,pre[jjj])
                print("cur", jjj,cur_best[jjj])
                #print("dernier", jjj,dernier_arrive[jjj])
            print(Q.total_nodes, "extracted", y,x)
            (a,t) = y
            P[a].add(t)
            (tpp,dis) = x
            for b in neighbors[a].keys():
                for (tp,edge) in neighbors[a][b]:
                    if tp >= t:
                        print("tp",tp)
                        print("salut2",(a,t),(b,tp))
                        self.relax_resting_paths_dij2(a,t,tp,pre,cur_best, events, events_rev, Q, nod)
                        print(cur_best)
                        self.relax_paper_dij2(a,b,t,tp,pre,cur_best, events, Q, nod, edge)
                        print("relax paths", cur_best)
        return (pre, cur_best)

    def until(self, events, events_rev):
        unt = dict()
        for v in self.nodes:
            unt[v] = dict()
            for i in range(0,len(self.node_presence[v]),2):
                j = events_rev[self.node_presence[v][i]]
                while j<len(events) and  events[j] <= self.node_presence[v][i+1]:
                    unt[v][events[j]] = self.node_presence[v][i+1]
                    j += 1
        return unt




    def dijkstra_temporal2_direc(self, s, events, events_rev, neighbors, d, neighbors_inv, unt):
        Q = fib.FibonacciHeap()
        cur_best = [ {t:(-numpy.Infinity,numpy.Infinity)   for t in events} for i in range(len(self.nodes)) ]
        pre = [{t:{}   for t in events} for i in range(len(self.nodes))]
        nod = dict()
        P = dict()
        for v in self.nodes:
            P[v] = set()
        for e in neighbors[s].keys():
            for j in range(0,len(self.link_presence[d[(s,e)]]),2):
            #for j in range(0,2,2):
                if self.link_presence[d[(s,e)]][j] != self.link_presence[d[(s,e)]][j+1]:
                    l = self.link_presence[d[(s,e)]][j:j+2]
                else:
                    l = self.link_presence[d[(s,e)]][j:j+1]
                for t in l:
                    cur_best[s][t] = (t,0)
                    pre[s][t] = {(0,0):(-1,-1)}
                    if (s,t) not in nod:
                        nod[s,t] = Q.insert( ((t - cur_best[s][t][0],cur_best[s][t][1]),(s,t) ) )
        print(Q.total_nodes)
        while Q.total_nodes != 0:
            print("nb_nodes", Q.total_nodes,"min",Q.find_min().data)
            (x,y) = Q.extract_min().data
            del nod[y]
            for jjj in self.nodes:
                print("pre", jjj,pre[jjj])
                print("cur", jjj,cur_best[jjj])
                #print("dernier", jjj,dernier_arrive[jjj])
            print(Q.total_nodes, "extracted", y,x)
            (a,t) = y
            P[a].add(t)
            #(tpp,dis) = x
            for b in neighbors_inv[a].keys():
                for (tp,edge) in neighbors_inv[a][b]:
                    if tp >= t and unt[a][t] >= tp:
                        print("tp_inv",tp)
                        print("salut2_inv",(a,t),(b,tp))
                        self.relax_resting_paths_dij2(a,t,tp,pre,cur_best, events, events_rev, Q, nod)
                        print("inv",cur_best)

            for b in neighbors[a].keys():
                for (tp,edge) in neighbors[a][b]:
                    if tp >= t and unt[a][t] >= tp:
                        print("tp",tp)
                        print("salut2",(a,t),(b,tp))
                        self.relax_resting_paths_dij2(a,t,tp,pre,cur_best, events, events_rev, Q, nod)
                        print(cur_best)
                        self.relax_paper_dij2(a,b,t,tp,pre,cur_best, events, Q, nod, edge)
                        print("relax paths", cur_best)
        return (pre, cur_best)


    def latencies(self,cur_best):
        latencies = [dict() for i in range(len(self.nodes))]
        last_depr = [dict() for i in range(len(self.nodes))]
        for k in self.nodes:
            for key in cur_best[k]:
                if cur_best[k][key][0] != -numpy.Infinity:
                    latencies[k][key] = cur_best[k][key]
                    if (cur_best[k][key][0] not in last_depr[k]):
                        last_depr[k][cur_best[k][key][0]] = key
                    else:
                        #we already have one element for the departure
                        if last_depr[k][cur_best[k][key][0]] > key:
                            del latencies[k][last_depr[k][cur_best[k][key][0]]]
                            last_depr[k][cur_best[k][key][0]] = key
                        else:
                            del latencies[k][key]

        return latencies

    def predecessor_graph(self,pre, node):
        G = nx.DiGraph()
        for k in self.nodes:
            if k != node:
                for key in pre[k].keys():
                    for v2 in pre[k][key].keys():
                        if v2[0] != node:
                            G.add_edge(v2,(k,key),interval=pre[k][key][v2])
                        else:
                            G.add_edge((node,0),(k,key),interval=pre[k][key][v2])
        return G



    def metapaths_from_predecessor(self,G):
        l = []
        paths = nx.all_simple_paths(G,(0,0.0))
        for path in map(nx.utils.pairwise, paths):
            for i in range(0,len(path)):
                if i == 0:
                    m = mw.Metawalk([],[])
                    m.add_link(e[0][0],e[1][0],G[e[0]][e[1]]['interval'])
            l.append(m)
        return l

    def depth_trav(self, G, node, last_inter, before_last_inter, sigma, depth, actual_poly, b, pow_actual, chevau, instantenous):
        instantenous2 = instantenous
        res = actual_poly[:]
        last_x, last_y = last_inter
        if depth == 1:
            if last_x != last_y:
                res[1] = numpy.around((last_y - last_x), decimals=5)
                # j'ai cru quil fallait mettre à un mais non je ne pense pas, car pour la suite il faut prendre le dernier temps de depart
                pow_actual = 1
                chevau = 1
            else:
                instantenous = False
                res[0] = 1
        else:
            if instantenous:
                blastx, blasty = before_last_inter
                if last_x == blastx and last_y == blasty:
                    pow_actual += 1
                    chevau += 1
                    #chevau ou pow_actual ne devrait rien changer ici
                    res[chevau] = numpy.around((last_y - last_x)/numpy.math.factorial(chevau), decimals=5)
                else:
                    instantenous = False
                    if last_x != last_y:
                        res[1] = numpy.around((last_y - last_x), decimals=5)
                        pow_actual = 1
                        chevau = 1
                        b = False
                    else:
                        pow_actual = 0
                        chevau = 0
                        res[0] = 1
            else:
                if last_x == last_y:
                    chevau = 0
                    if b == True:
                        res[0] = 1
                else:
                    pow_actual += 1
                    blastx, blasty = before_last_inter
                    b = False
                    if last_x == blastx and last_y == blasty:
                        chevau += 1
                        res[pow_actual] = res[pow_actual - 1] *numpy.around((last_y - last_x)/(chevau), decimals=5)
                        #+ res[pow_actual -1] - numpy.around((last_y - last_x)/numpy.math.factorial(chevau - 1), decimals=5)
                        res[pow_actual - 1] = 0
                    else:
                        chevau = 1
                        res[pow_actual] = res[pow_actual - 1] * numpy.around((last_y - last_x)/(chevau), decimals=5)
                        res[pow_actual - 1] = 0
        #add to sigma
        print("node", node)
        print("last",last_inter,"before_last",before_last_inter,"b",b,"instantenous",instantenous2)
        print("poly ",node,nppol.Polynomial(res))


        if node not in sigma:
            if instantenous == False:
                sigma[node] = (nppol.Polynomial(res),False,-1)
            else:
                sigma[node] = (nppol.Polynomial([1]),True,depth-1)
        else:
            (su, boolo, d) = sigma[node]
            sigma[node] = (su + nppol.Polynomial(res), boolo, d)


        # if res[0] == 1:
        #     res[0] = 0

        if instantenous:
            for e in G[node]:
                if G[node][e]['interval'][0] < last_inter[0]:
                    debut = G[node][e]['interval'][1]
                else:
                    debut = G[node][e]['interval'][0]
                link_mod = (debut, G[node][e]['interval'][1])
                self.depth_trav(G, e, link_mod, last_inter, sigma, depth + 1, actual_poly, b, pow_actual, chevau, instantenous2)
        else:
            for e in G[node]:
                if G[node][e]['interval'][0] < last_inter[0]:
                    debut = G[node][e]['interval'][1]
                else:
                    debut = G[node][e]['interval'][0]
                link_mod = (debut, G[node][e]['interval'][1])
                self.depth_trav(G, e, link_mod, last_inter, sigma, depth + 1,  res, b, pow_actual, chevau,  False)


    def trav_resting(self, G, e, last_inter, before_last_inter, pred_node, sigma, sigma_r, poly, depth, pred_depar, actual_depar):
        print("noder",e, "predr",pred_node, "last inter", last_inter, "before last inter", before_last_inter, "pred_depar", pred_depar, "actual_depar", actual_depar, sigma_r)
        # nodes are the same as before we add paths
        (su, boolo, d) = sigma[e]
        if pred_node[0] == e[0] and pred_depar == actual_depar:

            if e not in sigma_r:
                sigma_r[e] = (pred_node, su + sigma_r[pred_node][1], boolo, d)
            else:
                if pred_node > sigma_r[e][0]:
                    sigma_r[e] = (pred_node, su + sigma_r[pred_node][1], boolo, d)
        else:
            if e not in sigma_r:
                #no resting path here
                sigma_r[e] = ((-1,-1),su,boolo,d)



        visit = list(G[e])
        visit.sort()
        print("succ")
        dic_nodes = dict()
        for (x,y) in visit:
            if x in dic_nodes:
                dic_nodes[x].append(y)
            else:
                dic_nodes[x] = [y]
        for u in dic_nodes.keys():
            #normally it should be sorted
            for ii in range( 0, len(dic_nodes[u])):
                if ii == 0:
                    pred_node = (-1,-1)
                    pred_depar = -1
                else:
                    pred_node = (u,dic_nodes[u][ii-1])
                    pred_depar = actual_depar
                poly = [0 for jj in range(len(self.nodes))]
                self.trav_resting(G, (u,dic_nodes[u][ii]), G[e][(u,dic_nodes[u][ii])]['interval'], last_inter, pred_node, sigma, sigma_r, poly, depth + 1, pred_depar, actual_depar)
        # for ii in visit:
        #     print("next ",ii)
        # for i in range(0,len(visit)):
        #     if i == 0:
        #         pred = (-1,-1)
        #         pred_depar = -1
        #     else:
        #         pred = visit[i-1]
        #         pred_depar = pred[1]
        #     poly = [0 for i in range(len(self.nodes))]
        return


    def trav_bet_vol(self, x, G, e, vol_bet, pre):
        print("***** start *****", e)
        vv,tt = e
        if (vv in vol_bet) and (tt in vol_bet[vv]):
            return
        if vv not in vol_bet:
            vol_bet[vv] = dict()
        vol_bet[vv][tt] = dict()
        print("noder",e, "vv,tt", (vv,tt))
        visit = list(G[e])
        print("visit",visit)

        visit.sort()
        dic_nodes = dict()
        for (x,y) in visit:
            if x in dic_nodes:
                dic_nodes[x].append(y)
            else:
                dic_nodes[x] = [y]
        print("dic_nodes",dic_nodes)
        for u in dic_nodes.keys():
            l_ord = list(dic_nodes[u])
            l_ord.sort()
            #normally it should be sorted
            for ii in range( 0, len(l_ord)):
                print("u",u,"dic_nodes[u][ii]", dic_nodes[u][ii], "ii", ii, "l_ord[ii]",l_ord[ii], "l_ord[ii-1]",l_ord[ii-1])
                if ii == 0:
                    print("pre[u][l_ord[ii]]",pre[u][l_ord[ii]],"l_ord[ii]",l_ord[ii])
                    pred_node = (-1,-1)
                    vol_bet[vv][tt] = dict()
                    if vv != x:
                        t1, t2 = pre[u][l_ord[ii]][vv,tt]
                    else:
                        t1, t2 = pre[u][l_ord[ii]][vv,ll_ord[ii]]
                    if t1 == t2:
                        vol_bet[vv][tt][u,l_ord[ii]] = nppol.Polynomial([1])
                    else:
                        vol_bet[vv][tt][u,l_ord[ii]] = nppol.Polynomial([0,t2-t1])
                else:
                    #pred_node = (u,dic_nodes[u][ii-1])
                    print("pre[u][l_ord[ii]]",pre[u][l_ord[ii]],"l_ord[ii]",l_ord[ii])
                    if vv != x:
                        t1, t2 = pre[u][l_ord[ii]][vv,tt]
                    else:
                        t1, t2 = pre[u][l_ord[ii]][vv,ll_ord[ii]]
                    if t1 == t2:
                        vol_bet[vv][tt][u,l_ord[ii]] = vol_bet[vv][tt][u,l_ord[ii-1]]
                    else:
                        print("vol_bet[vv][tt]",vol_bet[vv][tt],"l_ord[ii]", l_ord[ii], "l_ord[ii-1]", l_ord[ii-1], "vv", vv, "tt", tt)
                        vol_bet[vv][tt][u,l_ord[ii]] = vol_bet[vv][tt][u,l_ord[ii-1]] + nppol.Polynomial([0,t2-t1])
                self.trav_bet_vol(x, G, (u,dic_nodes[u][ii]), vol_bet, pre)
                print("vol_bet[vv][tt]",vol_bet[vv][tt], "vv", vv, "tt", tt)
                print("vol_bet", vol_bet)
        print("**** end ****", "e", e)


    def trav_instant_graphs(self, G, e, GG, visited):
        if e in visited:
            return
        print("***** start *****", e)
        visit = list(G[e])
        print("visit",visit)
        visited.add(e)
        for (w,t_p) in visit:
            inter_act = G[e][w,t_p]['interval']
            t1,t2 = inter_act
            if t1 != t2 and t2 == e[1]:
                if inter_act not in GG:
                    GG[inter_act] = nx.DiGraph()
                GG[inter_act].add_edge(e,(w,t_p),weight=1)
            self.trav_instant_graphs(G,(w,t_p), GG, visited)

    def descendants_at_distance(self, G, source, distance):
        current_distance = 0
        current_layer = {(source,0)}
        visited = {(source,0)}

        # this is basically BFS, except that the current layer only stores the nodes at
        # current_distance from source at each iteration
        while current_distance < distance:
            next_layer = set()
            for node,dis in current_layer:
                for child in G[node]:
                    if child not in visited:
                        visited.add((child,dis+G[node][child]["weight"]))
                        next_layer.add((child,dis+G[node][child]["weight"]))
            current_layer = next_layer
            current_distance += 1

        return current_layer

    def transitive_closure_dag(self, G, topo_order=None):
        """variant from networkx"""
        if topo_order is None:
            topo_order = list(nx.topological_sort(G))

        TC = G.copy()

        # idea: traverse vertices following a reverse topological order, connecting
        # each vertex to its descendants at distance 2 as we go
        for v in reversed(topo_order):
            l = list(self.descendants_at_distance(TC, v, 2)) 
            for (e,d) in l:
                TC.add_edge(v,e,weight = d)
        return TC

    def sinks(self, G):
        return list((node for node, out_degree in G.out_degree() if out_degree == 0))

    def sources(self, G):
        return list((node for node, in_degree in G.in_degree() if in_degree == 0))

    def volume_between_direct_arrivals(self, x, G, pre):
        vol_bet = dict()
        #visited = [(x,0)]
        node = (x,0)
        vol_bet[x] = dict()
        vol_bet[x][0] = dict()
        for e in G[node]:
            self.trav_bet_vol(x, G, e, vol_bet, pre)
        return vol_bet

    def instant_metaedge_graphs(self, G):
        GG = dict()
        #visited = [(x,0)]
        l = self.sources(G)
        for node in l:
            for e in G[node]:
                self.trav_instant_graphs(G, e, GG, set())
        return GG

    def volume_metapaths(self, x, G):
        sigma = dict()
        #visited = [(x,0)]
        node = (x,0)
        for e in G[node]:
            poly = [0 for i in range(len(self.nodes))]
            self.depth_trav(G, e, G[node][e]['interval'], (-1,-1), sigma, 1, poly, True, 0, 0, True)
        return sigma



    def volume_metapaths_with_restingpaths(self, x, G, sigma):
        sigma_r = dict()
        node = (x,0)
        sigma_r[(x,0)] = ((-1,-1),nppol.Polynomial([0]))
        visit = list(G[node])
        visit.sort()
        dic_nodes = dict()
        for (x,y) in visit:
            if x in dic_nodes:
                dic_nodes[x].append(y)
            else:
                dic_nodes[x] = [y]
        for u in dic_nodes.keys():
            #normally it should be sorted
            for ii in range( 0, len(dic_nodes[u])):
                if ii == 0:
                    pred_node = (-1,-1)
                    actual_depar = dic_nodes[u][ii]
                    pred_depar = -1
                else:
                    pred_node = (u,dic_nodes[u][ii-1])
                    pred_depar = actual_depar
                    actual_depar = dic_nodes[u][ii]
                poly = [0 for jj in range(len(self.nodes))]
                self.trav_resting(G, (u,dic_nodes[u][ii]), G[node][(u,dic_nodes[u][ii])]['interval'], (-1,-1), pred_node, sigma, sigma_r, poly, 0, pred_depar, actual_depar)
        return sigma_r


    def volume_metapaths_with_restingpaths(self, x, G, sigma):
        sigma_r = dict()
        node = (x,0)
        sigma_r[(x,0)] = ((-1,-1),nppol.Polynomial([0]))
        visit = list(G[node])
        visit.sort()
        dic_nodes = dict()
        for (x,y) in visit:
            if x in dic_nodes:
                dic_nodes[x].append(y)
            else:
                dic_nodes[x] = [y]
        for u in dic_nodes.keys():
            #normally it should be sorted
            for ii in range( 0, len(dic_nodes[u])):
                if ii == 0:
                    pred_node = (-1,-1)
                    actual_depar = dic_nodes[u][ii]
                    pred_depar = -1
                else:
                    pred_node = (u,dic_nodes[u][ii-1])
                    pred_depar = actual_depar
                    actual_depar = dic_nodes[u][ii]
                poly = [0 for jj in range(len(self.nodes))]
                self.trav_resting(G, (u,dic_nodes[u][ii]), G[node][(u,dic_nodes[u][ii])]['interval'], (-1,-1), pred_node, sigma, sigma_r, poly, 0, pred_depar, actual_depar)
        return sigma_r


    # def trav_sigmastv(self, G, e, last_inter, before_last_inter, pred_node, pointer, pred_depar, actual_depar):
    #     print("noder",e, "predr",pred_node)
    #     # nodes are the same as before we add paths
    #     if pred_node[0] == e[0] and pred_depar == actual_depar:
    #         if e not in sigma_r:
    #             sigma_r[e] = (pred_node, sigma[e] + sigma_r[pred_node][1])
    #         else:
    #             if pred_node > sigma_r[e][0]:
    #                 sigma_r[e] = (pred_node, sigma[e] + sigma_r[pred_node][1])
    #     else:
    #         if e not in sigma_r:
    #             #no resting path here
    #             sigma_r[e] = ((-1,-1),sigma[e])

    #     visit = list(G[e])
    #     visit.sort()
    #     print("succ")
    #     for ii in visit:
    #         print("next ",ii)
    #     for i in range(0,len(visit)):
    #         if i == 0:
    #             pred = (-1,-1)
    #             pred_depar = -1
    #         else:
    #             pred = visit[i-1]
    #             pred_depar = pred[1]
    #         poly = [0 for i in range(len(self.nodes))]
    #         self.trav_resting(G, visit[i], G[e][visit[i]]['interval'], (-1,-1), pred, sigma, sigma_r, poly, depth + 1, pred_depar, visit[i][1])

    #     return


    #     def pointer_sigmastv(self, x, G):
    #     pointer = dict()
    #     node = (x,0)
    #     visit = list(G[node])
    #     visit.sort()
    #     for i in range(0,len(visit)):
    #         if i == 0:
    #             pred = (-1,-1)
    #             pred_depar = -1
    #         else:
    #             pred = visit[i-1]
    #             pred_depar = pred[1]
    #         poly = [0 for i in range(len(self.nodes))]
    #         self.sigmastv(G, visit[i], G[node][visit[i]]['interval'], (-1,-1), pred, pointer, pred_depar, visit[i][1])
    #     return pointer

    def zero_array(self, p):
        for e in p:
            if e != 0:
                return False
        return True

    def return_dict(self, e, d):
        if e in d:
            return d[e]
        return 0

    def pointers(self, sigma_r):
        pointer = dict()
        for k in self.nodes:
            l = list(self.event_times())
            l.sort()
            last = (-1,-1)
            for i in l:
                if (k,i) in sigma_r:
                    pointer[(k,i)] = (k,i)
                    last = (k,i)
                else:
                    pointer[(k,i)] = last
        return pointer

    def pointers3(self, cur_best):
        pointer = dict()
        for k in self.nodes:
            l = list(self.event_times())
            l.sort(reverse = True)
            after = (-1,-1)
            for i in l:
                if (k,i) in pointer:
                    pointer[(k,i)] = (k,i)
                    after = (k,i)
                else:
                    if after != (-1,-1):
                        if cur_best[k][after[0]] == cur_best[k][i]:
                            pointer[(k,i)] = last
                        else:
                            pointer[(k,i)] = (-1,-1)
                    else:
                        pointer[(k,i)] = after
        return pointer

    def pointers2(self, contri):
        pointer2 = dict()
        for k in self.nodes:
            l = list(self.event_times())
            l.sort()
            last = (-1,-1)
            for i in l:
                if i in contri[k]:
                    pointer2[(k,i)] = (k,i)
                    last = (k,i)
                else:
                    pointer2[(k,i)] = last
        return pointer2

    def latencies_rev(self, lat, events):
        res = [dict() for k in self.nodes]
        for k in self.nodes:
            for j in lat[k].keys():
                res[k][lat[k][j][0]] = j
            res[k][events[0]] = events[0]
        return res




    def delta_svvt(self, s, v, t, lat, contri, prev_next, sigma_r, pointer, pointer2, deltasvvt, lat_rev):
        if (v,t) in deltasvvt:
            return deltasvvt[(v,t)]
        print("call svvt, ","s",s,"v",v,"t",t)
        if s == v:
            return 0
        #voir papier matthieu clemence pour l'algo

        if t not in lat[v]:
            return 0
        prev = []
        next = []


        if t in prev_next[v]:
            prev = [lat[v][e][0] for e in prev_next[v][t] if e < t]
        t_contri = pointer2[(v,t)][1]
        #if t not in contri[v]:
        #    t_contri = pointer[(v,t)][1]

        t_sigma = pointer[(v,t)][1]
        #if (v,t) not in sigma_r:
        #    t_sigma = pointer[(v,t)][1]
        print("t_contri",v,t_contri)
        print("t_sigma",v,t_sigma)
        if t_contri == -1:
            return 0
        prev = [contri[v][t_contri][0]] + prev
        prev.sort(reverse = True)
        if t in prev_next[v]:
            next = [e for e in prev_next[v][t] if e > t]
        next = next + [contri[v][t_contri][1]]
        #prev.sort()
        print("prev", prev)
        print("next", next)

        left = 0
        right = 0

        contrib = 0
        s_prime = lat[v][t_contri][0]
        vol_tv = sigma_r[(v,t_sigma)][1]
        print("vol_tv",vol_tv, vol_tv.coef)
        if self.zero_array(vol_tv.coef):
            return 0
        for s_left in prev:
            # if (v,s_left) in sigma_r:
            #     left += sigma_r[pointer[(v,s_left)]][1]
            # else:
            #     left = 0

            a_prime = t
            right = 0
            for a_right in next:
                # if (v,a_right) in sigma_r:
                #     right += sigma_r[pointer[(v,a_right)]][1]
                # else:
                #     right = 0


                print("s_prime", s_prime, "s_left", s_left, "a_right", a_right, "a_prime", a_prime)
                tmp = (s_prime - s_left) * (a_right - a_prime) * vol_tv
                print("enum poly", tmp)
                enum_degree = self.actual_degree(tmp)
                print("actual enum", enum_degree)
                if enum_degree == -1:
                    enum = tuple([0])
                else:
                    enum = tuple([tmp.coef[enum_degree] if i == enum_degree else 0 for i in range(enum_degree+1)])



                print("left", left, "vol_tv", vol_tv, "right", right)
                tmp = left + vol_tv + right
                print("denum poly", tmp)
                denum_degree = self.actual_degree(tmp)
                print("actual denum", denum_degree)
                denum = tuple([tmp.coef[denum_degree] if i == denum_degree else 0 for i in range(denum_degree+1)])

                print("enum", enum, "denum", denum)

                res = nppol.polydiv(enum,denum)

                contrib += nppol.Polynomial(res[0])
                print("contrib",contrib)
                if (v,a_right) in sigma_r:
                    right += sigma_r[pointer[(v,a_right)]][1]
                # else:
                #     right = 0
                a_prime = a_right
            if (v,lat_rev[v][s_left]) in sigma_r:
                left += sigma_r[pointer[(v,lat_rev[v][s_left])]][1]
            # else:
            #     left = 0
            s_prime = s_left
        print("end svvt", contrib)
        deltasvvt[(v,t)] = contrib
        return contrib


    # def delta_svt(self, node, v, t, G, lat, contri, prev_next, sigma_r, pointer, pointer2, lat_rev):
    #     if node == v:
    #         return 0

    #     print("****** new_call, vt ********",(v,t))
    #     s = self.delta_svvt(node, v, t, lat, contri, prev_next, sigma_r, pointer, pointer2, {}, lat_rev)
    #     t_sigma = pointer[(v,t)][1]
    #     visit = list(G[(v,t_sigma)])
    #     for i in range(0,len(visit)):
    #         if visit[i][1] >= t:
    #             print("succ", visit[i])
    #             #svt = sigma_r[pointer[v,t]][1]
    #             svt = sigma_r[pointer[v,t]][1]
    #             swtp = sigma_r[pointer[visit[i]]][1]

    #             print("svt", svt)
    #             tmp = svt
    #             svt_degree = self.actual_degree(tmp)
    #             print("actual svt", svt_degree)
    #             svt_high = tuple([tmp.coef[svt_degree] if i == svt_degree else 0 for i in range(svt_degree+1)])
    #             if svt_high == ():
    #                     svt_hight = (0)

    #             print("svt_hight",svt_high)
    #             print("swtp", swtp)
    #             tmp = swtp
    #             swtp_degree = self.actual_degree(tmp)
    #             print("actual swtp", swtp_degree)
    #             swtp_high = tuple([tmp.coef[swtp_degree] if i == swtp_degree else 0 for i in range(swtp_degree+1)])
    #             print("swtp_high", swtp_high)
    #             if swtp_degree != -1:
    #                 res = nppol.polydiv(svt_high,swtp_high)
    #             else:
    #                 res = [0]
    #             print("division poly", res[0])
    #             s += res[0] * self.delta_svt(node, visit[i][0], visit[i][1], G, lat, contri, prev_next, sigma_r, pointer, pointer2, lat_rev)
    #     print("****** end_call, vt ********",(v,t), s)
    #     return s

    def polymul(self, v, w):
        if self.actual_degree(w) == -1 or self.actual_degree(v) == -1:
            return (0,0)
        else:
            degv = self.actual_degree(v)
            degw = self.actual_degree(w)
            vl = list(v.coef)
            wl = list(w.coef)
            if degv == 0 and vl[degv] == 0:
                return (0,0)
            if degw == 0 and wl[degw] == 0:
                return (0,0)
            return (vl[degv]*wl[degw], degv + degw)


    def polydiv(self, v, w):
        if self.actual_degree(w) == -1:
            return (0,0)
        else:
            degv = self.actual_degree(v)
            degw = self.actual_degree(w)
            vl = list(v.coef)
            wl = list(w.coef)
            if degv == -1:
                return (0,0)
            return (vl[degv]/wl[degw], degv - degw)

    def contri_delta_svvt(self, s, v, t, lat, contri, prev_next, sigma_r, deltasvvt, pointer, lat_rev):
        if (v,t) in deltasvvt:
            return deltasvvt[(v,t)]
        print("///////// call svvt, ","s",s,"v",v,"t",t)
        if s == v:
            return nppol.Polynomial([0])
        #voir papier matthieu clemence pour l'algo

        if t not in lat[v]:
            return nppol.Polynomial([0])
        prev = []
        next = []


        if t in prev_next[v]:
            prev = [lat[v][e][0] for e in prev_next[v][t] if e < t]
        #if t not in contri[v]:
        #    t = pointer[(v,t)][1]

        t_sigma = pointer[(v,t)][1]
        #if (v,t) not in sigma_r:
        #    t_sigma = pointer[(v,t)][1]
        print("t_contri",v,t)
        print("t_sigma",v,t_sigma)


        prev = [contri[v][t][0]] + prev
        prev.sort(reverse = True)
        if t in prev_next[v]:
            next = [e for e in prev_next[v][t] if e > t]
        next = next + [contri[v][t][1]]
        #prev.sort()
        print("prev", prev)
        print("next", next)

        left = 0
        right = 0

        contrib = 0
        s_prime = lat[v][t][0]
        vol_tv = sigma_r[(v,t_sigma)][1]
        print("vol_tv",vol_tv, vol_tv.coef)
        if self.zero_array(vol_tv.coef):
            return nppol.Polynomial([0])
        for s_left in prev:
            # if pointer[(v,s_left)] in sigma_r:
            #     left += sigma_r[pointer[(v,s_left)]][1]
            # else:
            #     left = 0

            a_prime = t
            right = 0
            for a_right in next:
                # if pointer[(v,a_right)] in sigma_r:
                #     right += sigma_r[pointer[(v,a_right)]][1]
                # else:
                #     right = 0

                print("s_prime", s_prime, "s_left", s_left, "a_right", a_right, "a_prime", a_prime)
                tmp = (s_prime - s_left) * (a_right - a_prime) * vol_tv
                print("enum poly", tmp)
                enum_degree = self.actual_degree(tmp)
                print("actual enum", enum_degree)
                enum = tmp
                # if enum_degree == -1:
                #     enum = tuple([0])
                # else:
                #     enum = tuple([tmp.coef[enum_degree] if i == enum_degree else 0 for i in range(enum_degree+1)])



                print("left", left, "vol_tv", vol_tv, "right", right)
                tmp = left + vol_tv + right
                print("denum poly", tmp)
                denum_degree = self.actual_degree(tmp)
                print("actual denum", denum_degree)
                denum = tmp
                #denum = tuple([tmp.coef[denum_degree] if i == denum_degree else 0 for i in range(denum_degree+1)])

                print("enum", enum, "denum", denum)
                print("enum-denum degree",enum_degree, denum_degree)

                res = self.polydiv(enum,denum)
                if res[1] < 0:
                    ress = [0]
                else:
                    ress =[0 for i in range(res[1]+1)]
                    ress[res[1]] = res[0]
                contrib += nppol.Polynomial(ress)
                print("contrib",contrib)
                if pointer[(v,a_right)] in sigma_r:
                    right += sigma_r[pointer[(v,a_right)]][1]
                # else:
                #     right = 0
                a_prime = a_right
            if pointer[(v,lat_rev[v][s_left])] in sigma_r:
                left += sigma_r[pointer[(v,lat_rev[v][s_left])]][1]
            # else:
            #     left = 0
            s_prime = s_left
        print("end svvt", contrib)
        deltasvvt[(v,t)] = contrib
        return contrib

    def coef_volume(self, x, v, t, w, t_p, sigma_r, pointer, pre, t1, t2, chev):
        print("divison_volume","v",v,"t",t,"w",w,"t_p",t_p, "t1", t1, "t2", t2, "chev", chev)
        if v == x :
            return nppol.Polynomial([1])
        # pv,pt = pointer[(v,t)]
        # #st1t2 = bet_vol[pv][pt][(w,t_p)]
        # (t1,t2) = pre[w][t_p][pv,pt]
        # (t1p,t2p) = pre[w][t_p][pv,pt]
        if t1 < t:
            (t1,t2) = (t2,t2)
        st1t2 = nppol.Polynomial([1])
        if t2 > t1:
            rr = [0 for ii in range(chev+1)]
            rr[chev] = math.pow((t2-t1),chev)/math.factorial(chev)
            st1t2 = nppol.Polynomial(rr)
        print("st1t2",st1t2)
        # if pointer[(v,t)] == (-1,-1) or pointer[(w,t_p)] == (-1,-1):
        #     return nppol.Polynomial([0])

        # #partie chevauchement
        # print("chevauchement")
        # for vvv in chevauchement:
        #     for ttt in chevauchement[vvv]:
        #         print(vvv,ttt,chevauchement[vvv][ttt])
        # if (w in chevauchement) and (t_p in chevauchement[w]):
        #     coef_chevau = chevauchement[w][t_p]
        #     if t1 == t2:
        #         coef_chevau = nppol.Polynomial([1])
        #     else:
        #         deg = self.actual_degree(coef_chevau)
        #         coef_chevau *= nppol.Polynomial([0,t2-t1])/(deg + 1)
        # else:
        #     if t1 == t2:
        #         coef_chevau = nppol.Polynomial([1])
        #     else:
        #         coef_chevau = nppol.Polynomial([0,t2-t1])
        # degg = self.actual_degree(coef_chevau)
        # if degg > 1:
        #     coeff = tuple(coef_chevau.coef)
        #     max_coeff = coeff[degg]
        #     coef_chevau = nppol.Polynomial([0,max_coeff])
        #     coef_chevau += st1t2
        # print("coef_chevau", coef_chevau, "st1t2", st1t2)
        svt = sigma_r[pointer[(v,t)]][1]
        swtp = sigma_r[pointer[(w,t_p)]][1]

        print("svt", svt)
        tmp = self.polymul(svt, st1t2)
        #tmp = tmp[0]
        print("svt*st1t2", tmp)
        svt_high = [0 for jj in range(tmp[1]+1)]
        svt_high[tmp[1]] = tmp[0]
        svt_high = nppol.Polynomial(svt_high)
        svt_degree = self.actual_degree(svt_high)
        print("actual svt*st1t2", svt_degree)
        
        #svt_high = tuple([tmp.coef[svt_degree] if i == svt_degree else 0 for i in range(svt_degree+1)])
        #if svt_high == ():
        #    svt_high = (0)
        print("svt_hight",svt_high)
        print("swtp", swtp)
        tmp = swtp
        swtp_degree = self.actual_degree(tmp)
        print("actual swtp", swtp_degree)
        swtp_high = tmp
        #swtp_high = tuple([tmp.coef[swtp_degree] if i == swtp_degree else 0 for i in range(swtp_degree+1)])
        print("swtp_high", swtp_high)
        if swtp_degree != -1:
            cof,puis = self.polydiv(svt_high,swtp_high)
            if puis < 0:
                res = [0]
            else:
                res = [0 for i in range(puis+1)]
                res[puis] = cof
        else:
            res = [0]
        print("res_div", res)
        return nppol.Polynomial(res)



    def contri_delta_svt(self, node, v, t, G, lat, contri, prev_next, sigma_r, partial_sum, contribution, deltasvvt, pointer, lat_rev, event, event_reverse, pre, GT):
        print("******** new call contri_delta_svt","v", v, "t", t)
        if (v not in contribution) or ((v in contribution) and (t not in contribution[v])):
            svvt = self.contri_delta_svvt(node, v, t, lat, contri, prev_next, sigma_r, deltasvvt, pointer, lat_rev)
            # normally only called on graph edges

            visit = list(G[(v,t)])
            dic_nodes_rev = dict()
            dic_nodes = dict()
            for (x,y) in visit:
                if y in dic_nodes_rev:
                    dic_nodes_rev[y].append(x)
                else:
                    dic_nodes_rev[y] = [x]
                if x in dic_nodes:
                    dic_nodes[x].append(y)
                else:
                    dic_nodes[x] = [y]
            print("dic_nodes", dic_nodes_rev)
            partial_sum = dict()
            # for u in dic_nodes:
            #     dic_nodes[u].sort()
                # if t not in dic_nodes[u]:
                #     dic_nodes[u] = [t] + dic_nodes[u]

            s = 0

            contrib_local = dict()
            l_ord = list(dic_nodes_rev.keys())
            l_ord.sort()
            for ii in range(len(l_ord)-1,-1,-1):
                #normally it should be sorted
                for u in dic_nodes_rev[l_ord[ii]]:
                    # svt = sigma_r[(v,t)][1]
                    # swtp = sigma_r[(u,dic_nodes[u][ii])][1]

                    # print("svt", svt)
                    # tmp = svt
                    # svt_degree = self.actual_degree(tmp)
                    # print("actual svt", svt_degree)
                    # svt_high = tuple([tmp.coef[svt_degree] if i == svt_degree else 0 for i in range(svt_degree+1)])
                    # if svt_high == ():
                    #     svt_high = (0)
                    # print("svt_hight",svt_high)
                    # print("swtp", swtp)
                    # tmp = swtp
                    # swtp_degree = self.actual_degree(tmp)
                    # print("actual swtp", swtp_degree)
                    # swtp_high = tuple([tmp.coef[swtp_degree] if i == swtp_degree else 0 for i in range(swtp_degree+1)])
                    # print("swtp_high", swtp_high)
                    # if swtp_degree != -1:
                    #     res = nppol.polydiv(svt_high,swtp_high)
                    # else:
                    #     res = [0]
                    w,t_p = (u,l_ord[ii])

                    print("(w,t')",(w,t_p))
                    self.contri_delta_svt(node, w, t_p, G, lat, contri, prev_next, sigma_r, partial_sum, contribution, deltasvvt, pointer, lat_rev, event, event_reverse, pre, GT)

                    (t1,t2) = pre[w][t_p][v,t]
                    if t1 != t2:
                        for yp,tpp in GT[t1,t2][w,t_p]:
                            print("*!*!*!*!*!*!  instant graph","v",v,"t",t,"t1",t1,"t2",t2, "w",w,"t_p",t_p, "yp", yp, "tpp", tpp)
                            self.contri_delta_svt(node, yp, tpp, G, lat, contri, prev_next, sigma_r, partial_sum, contribution, deltasvvt, pointer, lat_rev, event, event_reverse, pre, GT)
                            res2 = self.coef_volume(node, v, t, yp, tpp, sigma_r, pointer, pre, t1, t2, GT[t1,t2][w,t_p][yp,tpp]["weight"]+1)
                            s += res2 * contribution[yp][tpp]


                    res = self.coef_volume(node, v, t, u, l_ord[ii], sigma_r, pointer, pre, t1, t2, 1)
                    #appel recursif
                    s += res * contribution[w][t_p]
                    if l_ord[ii] not in partial_sum:
                        partial_sum[l_ord[ii]] = s
                    else:
                        partial_sum[l_ord[ii]] += s

                    print("******** half call contri_delta_svt","v", v, "t", t, "sum", s)
                    if ii != 0:
                        jj = event_reverse[l_ord[ii-1]]
                    else:
                        jj = event_reverse[t]
                    print("u", u, "dic_nodes[u]", dic_nodes[u])
                    print("******** half after call contri_delta_svt","v", v, "t", t)
                    print("dic_nodes[u]",dic_nodes[u],"u",u)
                    if True:#ii != len(dic_nodes[u])-1:
                        for jjj in range(jj+1,event_reverse[l_ord[ii]]+1):
                            #if not (j==0 and jjj==jj):

                            print("v", v, "t",t,"event[jj]", event[jj], "dic_nodes[u][ii]",l_ord[ii], "index actual event",jj,"index succ events", jjj, "contri event time" ,event[jjj])
                            print("comp vol", sigma_r[pointer[v,t]][1], sigma_r[pointer[v,event[jjj]]][1])

                            if v not in contrib_local:
                                contrib_local[v] = dict()
                            if not (v in contribution and event[jjj] in contribution[v]):
                                if sigma_r[pointer[v,t]][1] != sigma_r[pointer[v,event[jjj]]][1]:
                                    print("ERREUUUUUUUR ",sigma_r[pointer[v,t]][1],sigma_r[pointer[v,event[jjj]]][1])
                                #if event[jjj] in contrib_local[v]:
                                print("add_contri_local","v",v,"event[jjj]",event[jjj],"w,t_p",w,t_p)
                                    #contrib_local[v][event[jjj]] += contribution[w][t_p]*(self.coef_volume(node, v,event[jjj],w,t_p,sigma_r, pointer, pre))
                                if event[jjj] != t_p:
                                    print("ici")
                                    contrib_local[v][event[jjj]] = partial_sum[l_ord[ii]]
                                else:
                                    print("la")
                                    if ii == len(l_ord)-1:
                                        contrib_local[v][event[jjj]] =  contribution[w][t_p]*(self.coef_volume(node, v,event[jjj],w,t_p,sigma_r, pointer, pre, t2, t1, 1))
                                    else:
                                        contrib_local[v][event[jjj]] = partial_sum[l_ord[ii+1]]  + contribution[w][t_p]*(self.coef_volume(node, v,event[jjj],w,t_p,sigma_r, pointer, pre, t2, t1, 1))
                                print("contrib_local[v][event[jjj]]",contrib_local[v][event[jjj]],"partial_sum[l_ord[ii]]",partial_sum[l_ord[ii]],"contribution[w][t_p]",contribution[w][t_p],"res",res)
                                # else:
                                #     print("add_contri_local","v",v,"event[jjj]",event[jjj],"first w,t_p",w,t_p)
                                #     if event[jjj] != t_p:
                                #         contrib_local[v][event[jjj]] = partial_sum[dic_nodes[u][ii]]
                                #     else:
                                    #contrib_local[v][event[jjj]] = contribution[w][t_p]*(self.coef_volume(node, v,event[jjj],w,t_p,sigma_r, pointer, pre))
                    # else:
                    #     for jjj in range(jj+1,len(event_reverse)):
                    #         if not (j==0 and jjj==jj):
                    #         contribution[v][event[jjj]] = s*(self.coef_volume(v,event[jjj],w,t_p,sigma_r, pointer))


                    # if v not in contribution:
                    #     contribution[v] = dict()
                    # contribution[v][dic_nodes[u][ii]] = s
                    # print("chevauchement_contri")
                    # for vvv in chevauchement:
                    #     for ttt in chevauchement[vvv]:
                    #         print(vvv,ttt,chevauchement[vvv][ttt])
                    # t1,t2 = pre[u][l_ord[ii]][v,t]
                    # if len(visit) == 0 or not ((w in chevauchement) and (t_p in chevauchement[w])):
                    #     if t1 == t2:
                    #         coef_chevau = nppol.Polynomial([1])
                    #     else:
                    #         coef_chevau = nppol.Polynomial([0,t2-t1])
                    #     if v not in chevauchement:
                    #         chevauchement[v] = dict()
                    #     chevauchement[v][t_p] = coef_chevau

                    # else:
                    #     coef_chevau = chevauchement[w][t_p]
                    #     if (v not in chevauchement):
                    #         chevauchement[v] = dict()
                    #     if t1 == t2 :
                    #         coef_chevau =  nppol.Polynomial([1])
                    #     else:
                    #         deg = self.actual_degree(coef_chevau)
                    #         coef_chevau *= nppol.Polynomial([0,t2-t1])/(deg + 1)
                    #     chevauchement[v][t_p] = coef_chevau

            if v not in contribution:
                contribution[v] = dict()
            for vv in contrib_local:
                for ss in contrib_local[vv]:
                    contribution[vv][ss] = contrib_local[vv][ss]
            
            contribution[v][t] = s + svvt
        print("******** end call contri_delta_svt","v", v, "t", t,"contribution[v][t]",contribution[v][t])
        return contribution, partial_sum


    def closest_arrival_contri(self, cur_best, events):
        res = [ dict() for k in self.nodes ]
        for k in self.nodes:
            last_depar = -1
            last_arrival = -1
            for i in events:
                if cur_best[k][i][0] != - numpy.Infinity:
                    dep = cur_best[k][i][0]
                    res[k][i] = last_arrival
                    if last_depar != dep:
                        dep = last_depar
                        last_arrival = i
                        #last_depar = dep
        return res



    def contri_link_stream(self, contribution, partial_sum, event, event_reverse, close_arrival, sigma_r, pointer):
        contri_finale = [dict() for k in self.nodes]
        for k in self.nodes:
            if k in contribution:
                for i in contribution[k]:
                    contri_finale[k][i] = contribution[k][i]
                    times = partial_sum[(k,i)].keys()
                    times = list(times)
                    times.reverse()
                    times = [i] + times
                    print("nodes",k,"i",i,"partial_times",times)
                    for j in range(0,len(times) -1):
                        jj = event_reverse[times[j]]
                        print("jj",jj,times[j+1],event_reverse[times[j+1]])
                        for jjj in range(jj+1,event_reverse[times[j+1]]+1):
                            #if not (j==0 and jjj==jj):
                            contri_finale[k][event[jjj]] = partial_sum[(k,i)][times[j+1]]*(self.coef_volume(k,event[jjj],k,i,sigma_r, pointer))
        return contri_finale




               # if (self.actual_degree(sigma_r[(node,latencies[j][0])][1]) > 0) or (self.actual_degree(sigma_r[(node,latencies[i][0])][1]) > 0):

    def cal_lat(self, arr,latencies):
        #return latency
        return arr - latencies[arr][0]

    def check_contri(self, j, i,latencies, sigma_r, rev_events, node):
        lati = (self.cal_lat(i,latencies),latencies[i][1])
        latj = (self.cal_lat(j,latencies),latencies[j][1])
        if (lati[0] == 0 and latj[0] == 0) and (lati[1] == latj[1]) and lati[1] != 0:
            # latencies are instantenous
            if numpy.abs(rev_events[j] - rev_events[i]) == 1:
                #latencies are successive
                print("lat[i]",latencies[i],"latj[j]",latencies[j],"node",node)

                if sigma_r[(node,latencies[j][0])][2] == True or sigma_r[(node,latencies[i][0])][2] == True:
                    print("ok = > lat[i]",latencies[i],"latj[j]",latencies[j],"node",node)
                    return 2
            else:
                return 1
        if latj < lati:
            return 1
        elif latj == lati:
            return 0
        else:
            return -1

    def contribution_each_latency(self,latencies, sigma_r, rev_events):
        maxi = max(self.times)
        #        contri = [dict() for i in range(len(self.nodes))]
        contri = [dict() for i in range(len(self.nodes))]
        prev_next = [dict() for i in range(len(self.nodes))]
        for k in self.nodes:
            # l contains first arrival times
            l = [e for e in latencies[k].keys() ]
            l.sort()
            for i in range(0,len(l)):
                #check left contribution
                j = i - 1
                S = -1
                b = True
                while(j >= 0 and b):
                    cond = self.check_contri(l[j], l[i], latencies[k], sigma_r, rev_events, k)
                    if cond == 2:
                        S = latencies[k][l[i]][0]
                        b = False
                    if  cond == 1:
                        S = latencies[k][l[j]][0]
                        b = False
                    else:
                        if cond == 0:
                            if l[i] not in prev_next[k]:
                                prev_next[k][l[i]] = [l[j]]
                            else:
                                prev_next[k][l[i]].append(l[j])
                        j = j - 1
                if S == -1:
                    S = 0
                #check right contribution
                j = i + 1
                A = -1
                b = True
                while(j < len(l) and b):
                    cond = self.check_contri(l[j], l[i], latencies[k], sigma_r, rev_events, k)
                    if cond == 2:
                        A = l[i]
                        b = False
                    if cond == 1:
                        A = l[j]
                        b = False
                    else:
                        if cond == 0:
                            if l[i] not in prev_next[k]:
                                prev_next[k][l[i]] = [l[j]]
                            else:
                                prev_next[k][l[i]].append(l[j])
                        j = j + 1
                if A == -1:
                    A = maxi
                contri[k][l[i]] = (S,A)
        return contri,prev_next

    def volume_sz_t_v(self, G, latencies, contri, z, v, t):
        lat = -1
        for e in latencies[z].keys():
            if t >= e and t <= latencies[z][e] :
                lat = e
        paths = nx.all_simple_paths(G,(0,0.0),(z,latencies[z][lat]))



#arrival times are unique, so we can define cur_best[b][alpha], is it possible to put the metaedge in pre?
    def fastest_paths_from_vertex(self,x,boo):
                             #b is bool for new and old version 0 for old and 1 for new, to be removed
        """
        the function returns all metapaths in a link stream
        and is based on bellman-ford algorithm

        :param x: x the node from which we want the metawalks
        :return: a list of the metapaths in the link stream 
        """
        pre = dict()
        cur_best = dict()

        final_paths = [dict() for i in range(len(self.nodes))]
        tmp_paths = [dict() for i in range(len(self.nodes))]

        #the 2 loops inside the first are to iterate over each temporal link 
        for k in self.nodes:
            for i in range(0,len(self.links)):
                a,b = self.links[i]
                for j in range(0,len(self.link_presence[i]),2):
                    t1,t2 = self.link_presence[i][j:j+2]
                    if boo == 0:
                        self.check_edge(x,a,b,t1,t2,final_paths,tmp_paths)
                    else:
                        #add commented line for both directions of edges
                        self.extend_paths(x,a,b,t1,t2,final_paths, tmp_paths)
                        self.extend_paths(x,b,a,t1,t2,final_paths, tmp_paths)

        return final_paths

    #############################################################################
    #                       discrete                                            #
    #############################################################################

    def predecessor_graph_dis(self,pre, node):
        G = nx.DiGraph()
        for k in self.nodes:
            if k != node:
                for key in pre[k].keys():
                    for v2 in pre[k][key].keys():
                        v,t = v2
                        G.add_edge((v,t),(k,key),interval=pre[k][key][v2][0])
                        #G.add_edge((self.node_to_label[v],t),(self.node_to_label[k],key),interval=pre[k][key][v2][0])
        return G

    def graph_to_ordered(self, G, ev, ev_rev):
        return org.OrdGraphDis(G.nodes, G.edges, ev, ev_rev, self.sinks(G))

    def cur_best_to_array(self, cur_best, ev, ev_rev):
        cur_b_arr = [ [0 for t in ev    ]   for k in self.nodes]
        for v in self.nodes:
            for t in cur_best[v]:
                cur_b_arr[v][ev_rev[t]] = (t,cur_best[v][t][0],cur_best[v][t][1])
        return cur_b_arr

    def latencies_dis(self,cur_b_arr, ev, ev_rev):
        latencies = [ [0 for t in ev    ]   for k in self.nodes]
        last_depr = [ [numpy.Infinity for t in ev    ]   for k in self.nodes]
        for k in self.nodes:
            for i in range(0,len(cur_b_arr[k])):
                if cur_b_arr[k][i][1] != -numpy.Infinity:
                    if latencies[k][ev_rev[cur_b_arr[k][i][1]]] == 0:
                        latencies[k][ev_rev[cur_b_arr[k][i][1]]] = (cur_b_arr[k][i][0],cur_b_arr[k][i][2])
                    if last_depr[k][ev_rev[cur_b_arr[k][i][1]]] == numpy.Infinity:
                        last_depr[k][ev_rev[cur_b_arr[k][i][1]]] = cur_b_arr[k][i][0]
                    else:
                        #we already have one element for the departure
                        if last_depr[k][ev_rev[cur_b_arr[k][i][1]]] > cur_b_arr[k][i][0]:
                            last_depr[k][ev_rev[cur_b_arr[k][i][1]]] = cur_b_arr[k][i][0]
                            #print("i",i,"cur_b_arr[k][i][0]",cur_b_arr[k][i][0],"ev_rev[cur_b_arr[k][i][1]]",ev_rev[cur_b_arr[k][i][1]])
                            latencies[k][ev_rev[cur_b_arr[k][i][1]]] = (cur_b_arr[k][i][0],cur_b_arr[k][i][2])

        return latencies

    def latencies_without_0_and_rev(self, lat, ev):
        latency = [[] for k in self.nodes]
        latency_rev = [[] for k in self.nodes]
        for k in self.nodes:
            for i in range(0,len(ev)):
                if lat[k][i] != 0:
                    latency[k].append([ev[i],lat[k][i][0], lat[k][i][1]])
                    latency_rev[k].append([ lat[k][i][0],ev[i],lat[k][i][1] ])
        return latency, latency_rev






    # def trav_resting_dis(self, G, e, last_inter, before_last_inter, pred_node, sigma, sigma_r, poly, depth, pred_depar, actual_depar):
    #     print("noder",e, "predr",pred_node, "last inter", last_inter, "before last inter", before_last_inter, "pred_depar", pred_depar, "actual_depar", actual_depar, sigma_r)
    #     # nodes are the same as before we add paths
    #     su = sigma[e]
    #     if pred_node[0] == e[0] and pred_depar == actual_depar:

    #         if e not in sigma_r:
    #             sigma_r[e] = (pred_node, su + sigma_r[pred_node][1])
    #         else:
    #             if pred_node > sigma_r[e][0]:
    #                 sigma_r[e] = (pred_node, su + sigma_r[pred_node][1])
    #     else:
    #         if e not in sigma_r:
    #             #no resting path here
    #             sigma_r[e] = ((-1,-1),su)



    #     visit = list(G[e])
    #     visit.sort()
    #     print("succ")
    #     dic_nodes = dict()
    #     for (x,y) in visit:
    #         if x in dic_nodes:
    #             dic_nodes[x].append(y)
    #         else:
    #             dic_nodes[x] = [y]
    #     for u in dic_nodes.keys():
    #         #normally it should be sorted
    #         for ii in range( 0, len(dic_nodes[u])):
    #             if ii == 0:
    #                 pred_node = (-1,-1)
    #                 pred_depar = -1
    #             else:
    #                 pred_node = (u,dic_nodes[u][ii-1])
    #                 pred_depar = actual_depar
    #             poly = [0 for jj in range(len(self.nodes))]
    #             self.trav_resting_dis(G, (u,dic_nodes[u][ii]), G[e][(u,dic_nodes[u][ii])]['interval'], last_inter, pred_node, sigma, sigma_r, poly, depth + 1, pred_depar, actual_depar)
    #     return

    def first_edge_rec(self, e, edge, f_edge, G):
        if e in f_edge:
            return
        f_edge[e] = edge
        l = list(G[e])
        for ee in l:
            self.first_edge_rec(ee, edge, f_edge, G)

    def dictionary_first_edge(self, G):
        f_edge = dict()
        source = self.sources(G)
        for sou in source:
            f_edge[sou] = sou[1]
            for e in list(G[sou]):
                self.first_edge_rec(e, sou[1], f_edge, G)
        return f_edge

    def optimal_with_resting_dis(self, node, f_edge, events, G, sigma):
        sigma_r = dict()
        for k in self.nodes:
            pred = -1
            for t in events:
                if k == node:
                    sigma_r[(k,t)] = 0
                else:
                    if pred == -1:
                        if (k,t) in G:
                            sigma_r[(k,t)] = sigma[(k,t)]
                            pred = t
                            edge = f_edge[(k,t)]
                        else:
                            sigma_r[(k,t)] = 0
                    else:
                        if (k,t) in G:
                            edge2 = f_edge[(k,t)]
                            if edge == edge2:
                                sigma_r[(k,t)] = sigma_r[(k,pred)] + sigma[(k,t)]
                                pred = t
                            else:
                                sigma_r[(k,t)] = sigma[(k,t)]
                                pred = t
                                edge = f_edge[(k,t)]
                        else:
                            sigma_r[(k,t)] = sigma_r[(k,pred)]
        return sigma_r



    def vol_rec(self, s, e, G_rev, sigma):
        if e in sigma:
            return
        l = list(G_rev[e])
        print(e,l,sigma)
        if len(l) == 1:
            w,tp = l[0]
            if w == s:
                sigma[e] = 1
            else:
                self.vol_rec(s, (w,tp), G_rev, sigma)
                sigma[e] = sigma[(w,tp)]
        else:
            res = 0
            for (w,tp) in l:
                self.vol_rec(s, (w,tp), G_rev, sigma)
                res += sigma[(w,tp)]

            sigma[e] = res



    def volume_metapaths_dis(self, G, s):
        sigma = dict()
        sink = self.sinks(G)
        G_rev = G.reverse(copy=True)
        for e in sink:
            self.vol_rec(s, e, G_rev, sigma)
        return sigma

    # def volume_metapaths_with_restingpaths_dis(self, G, sigma):
    #     sigma_r = dict()
    #     source = list(self.sources(G))
    #     for (v,t) in source:
    #         sigma_r[(v,t)] = ((-1,-1),0)
    #         visit = list(G[(v,t)])
    #         visit.sort()
    #         dic_nodes = dict()
    #         for (x,y) in visit:
    #             if x in dic_nodes:
    #                 dic_nodes[x].append(y)
    #             else:
    #                 dic_nodes[x] = [y]
    #         for u in dic_nodes.keys():
    #             #normally it should be sorted
    #             for ii in range( 0, len(dic_nodes[u])):
    #                 if ii == 0:
    #                     pred_node = (-1,-1)
    #                     actual_depar = dic_nodes[u][ii]
    #                     pred_depar = -1
    #                 else:
    #                     pred_node = (u,dic_nodes[u][ii-1])
    #                     pred_depar = actual_depar
    #                     actual_depar = dic_nodes[u][ii]
    #                 poly = [0 for jj in range(len(self.nodes))]
    #                 self.trav_resting_dis(G, (u,dic_nodes[u][ii]), G[(v,t)][(u,dic_nodes[u][ii])]['interval'], (-1,-1), pred_node, sigma, sigma_r, poly, 0, pred_depar, actual_depar)
    #     return sigma_r

    def cal_lat_dis(self, arr,latencies):
        #return latency
        return latencies[arr][0] - latencies[arr][1]


    def check_contri_dis(self, j, i,latencies):
        lati = (self.cal_lat_dis(i,latencies),latencies[i][2])
        latj = (self.cal_lat_dis(j,latencies),latencies[j][2])
        if latj < lati:
            return 1
        elif latj == lati:
            return 0
        else:
            return -1

    def coef_volume_dis(self, x, v, t, w, t_p, sigma_r):
        print("divison_volume","v",v,"t",t,"w",w,"t_p",t_p)
        if v == x :
            return 1
        svt = sigma_r[(v,t)]
        swtp = sigma_r[(w,t_p)]
        print("svt", svt)
        print("swtp", swtp)
        if svt == 0:
            return 0
        return (svt/swtp)

    def lat_to_dic(self, lat):
        latency = [ dict()  for k in self.nodes]
        latency_rev = [ dict()  for k in self.nodes]
        for k in self.nodes:
            for (x,y,z) in lat[k]:
                latency[k][y] = (x,z)
                latency_rev[k][x] = y
        return latency, latency_rev

    def contribution_each_latency_dis(self,latencies, mini, maxi):
        #        contri = [dict() for i in range(len(self.nodes))]
        contri = [dict() for i in range(len(self.nodes))]
        prev_next = [dict() for i in range(len(self.nodes))]
        for k in self.nodes:
            # l contains first arrival times
            #l = [e for e in latencies[k].keys() ]
            #l.sort()
            l = [x for (x,y,z) in latencies[k] ]
            for i in range(0,len(l)):
                #check left contribution
                j = i - 1
                S = -numpy.Infinity
                b = True
                while(j >= 0 and b):
                    #cond = self.check_contri_dis(l[j], l[i], latencies[k], k)
                    cond = self.check_contri_dis(j, i, latencies[k])
                    if  cond == 1:
                        #S = latencies[k][l[j]][0]
                        S = latencies[k][j][1]
                        b = False
                    else:
                        if cond == 0:
                            if l[i] not in prev_next[k]:
                                prev_next[k][l[i]] = [l[j]]
                            else:
                                prev_next[k][l[i]].append(l[j])
                        j = j - 1
                if S == -numpy.Infinity:
                    S = mini
                #check right contribution
                j = i + 1
                A = -numpy.Infinity
                b = True
                while(j < len(l) and b):
                    #cond = self.check_contri_dis(l[j], l[i], latencies[k], k)
                    cond = self.check_contri_dis(j, i, latencies[k])
                    if cond == 1:
                        A = l[j]
                        b = False
                    else:
                        if cond == 0:
                            if l[i] not in prev_next[k]:
                                prev_next[k][l[i]] = [l[j]]
                            else:
                                prev_next[k][l[i]].append(l[j])
                        j = j + 1
                if A == -numpy.Infinity:
                    A = maxi
                contri[k][l[i]] = (S,A)
        return contri,prev_next

    def contri_delta_svvt_dis(self, s, v, t, lat, contri, prev_next, sigma_r, lat_rev):
        #if (v,t) in deltasvvt:
            #return deltasvvt[(v,t)]
        print("///////// call svvt, ","s",s,"v",v,"t",t)
        if s == v:
            return 0
        #voir papier matthieu clemence pour l'algo

        if t not in lat[v]:
            return 0
        prev = []
        next = []


        if t in prev_next[v]:
            prev = [lat[v][e][0] for e in prev_next[v][t] if e < t]
            next = [e for e in prev_next[v][t] if e > t]
        #if t not in contri[v]:
        #    t = pointer[(v,t)][1]

        #t_sigma = pointer[(v,t)][1]
        #if (v,t) not in sigma_r:
        #    t_sigma = pointer[(v,t)][1]
        #print("t_contri",v,t)
        #print("t_sigma",v,t_sigma)


        #prev.sort(reverse = True)
        #prev.reverse()
        prev =  prev + [contri[v][t][0]]
        next = next + [contri[v][t][1]]
        #prev.sort()
        print("prev", prev)
        print("next", next)

        left = 0
        right = 0

        contrib = 0
        s_prime = lat[v][t][0]
        vol_tv = sigma_r[(v,t)]
        # print("vol_tv",vol_tv, vol_tv.coef)
        # if self.zero_array(vol_tv.coef):
        #     return nppol.Polynomial([0])
        for s_left in prev:
            # if pointer[(v,s_left)] in sigma_r:
            #     left += sigma_r[pointer[(v,s_left)]][1]
            # else:
            #     left = 0

            a_prime = t
            right = 0
            for a_right in next:
                # if pointer[(v,a_right)] in sigma_r:
                #     right += sigma_r[pointer[(v,a_right)]][1]
                # else:
                #     right = 0

                print("s_prime", s_prime, "s_left", s_left, "a_right", a_right, "a_prime", a_prime)
                enum = (s_prime - s_left) * (a_right - a_prime) * vol_tv
                # print("enum poly", tmp)
                # enum_degree = self.actual_degree(tmp)
                # print("actual enum", enum_degree)
                # enum = tmp
                # if enum_degree == -1:
                #     enum = tuple([0])
                # else:
                #     enum = tuple([tmp.coef[enum_degree] if i == enum_degree else 0 for i in range(enum_degree+1)])



                print("left", left, "vol_tv", vol_tv, "right", right)
                denum = left + vol_tv + right
                # print("denum poly", tmp)
                # denum_degree = self.actual_degree(tmp)
                # print("actual denum", denum_degree)
                # denum = tmp
                #denum = tuple([tmp.coef[denum_degree] if i == denum_degree else 0 for i in range(denum_degree+1)])

                print("enum", enum, "denum", denum)
                #print("enum-denum degree",enum_degree, denum_degree)

                # res = self.polydiv(enum,denum)
                # if res[1] < 0:
                #     ress = [0]
                # else:
                #     ress =[0 for i in range(res[1]+1)]
                #     ress[res[1]] = res[0]
                contrib += (enum/denum)
                print("contrib",contrib)
                #if pointer[(v,a_right)] in sigma_r:
                if a_right in lat[v]:
                    right += sigma_r[(v,a_right)]
                # else:
                #     right = 0
                a_prime = a_right
            #if pointer[(v,lat_rev[v][s_left])] in sigma_r:
            if (s_left in lat_rev[v]) and (lat_rev[v][s_left] in lat[v]):
            #if lat_rev[v][s_left] in lat[v]:
                left += sigma_r[(v,lat_rev[v][s_left])]
            # else:
            #     left = 0
            s_prime = s_left
        print("end svvt", contrib)
        #deltasvvt[(v,t)] = contrib
        return contrib


    def contri_delta_svt_dis(self, node, v, t, l_nei, sigma_r, contribution, deltasvvt, event, event_reverse):
        print("******** new call contri_delta_svt","v", v, "t", t)
        if (v not in contribution) or ((v in contribution) and (t not in contribution[v])):
            partial_sum = dict()
            s = 0
            contrib_local = dict()
            # l_ord = list(dic_nodes_rev.keys())
            # l_ord.sort()
            for ii in range(len(l_nei[(v,t)])-1,-1,-1):
                #normally it should be sorted
                for u in l_nei[v,t][ii][1]:
                    w,t_p = (u,l_nei[v,t][ii][0])
                    print("(w,t')",(w,t_p))
                    self.contri_delta_svt_dis(node, w, t_p, l_nei, sigma_r, contribution, deltasvvt, event, event_reverse)
                    res = self.coef_volume_dis(node, v, t, w, t_p, sigma_r)
                    #appel recursif
                    s += res * contribution[w][t_p]
                    if l_nei[v,t][ii][0] not in partial_sum:
                        partial_sum[l_nei[v,t][ii][0]] = s
                    else:
                        partial_sum[l_nei[v,t][ii][0]] += s

                    print("******** half call contri_delta_svt","v", v, "t", t, "sum", s)
                    if ii != 0:
                        jj = event_reverse[l_nei[v,t][ii-1][0]]
                    else:
                        jj = event_reverse[t]
                    print("u", u, "dic_nodes[u]", l_nei[v,t][ii])
                    print("******** half after call contri_delta_svt","v", v, "t", t)
                    print("dic_nodes[u]",l_nei[v,t],"u",u)
                    if True:#ii != len(dic_nodes[u])-1:
                        for jjj in range(jj+1,event_reverse[l_nei[v,t][ii][0]]+1):
                            print("v", v, "t",t,"event[jj]", event[jj], "dic_nodes[u][ii]", "index actual event",jj,"index succ events", jjj, "contri event time" ,event[jjj])
                            print("comp vol", sigma_r[v,t], sigma_r[v,event[jjj]])

                            if v not in contrib_local:
                                contrib_local[v] = dict()
                            if not (v in contribution and event[jjj] in contribution[v]):
                                if sigma_r[v,t] != sigma_r[v,event[jjj]]:
                                    print("ERREUUUUUUUR ",sigma_r[v,t],sigma_r[v,event[jjj]])
                                print("add_contri_local","v",v,"event[jjj]",event[jjj],"w,t_p",w,t_p)
                                if event[jjj] != t_p:
                                    print("ici")
                                    contrib_local[v][event[jjj]] = partial_sum[l_nei[v,t][ii][0]]
                                else:
                                    print("la")
                                    if ii == len(l_nei[v,t])-1:
                                        contrib_local[v][event[jjj]] =  contribution[w][t_p]*(self.coef_volume_dis(node, v,event[jjj],w,t_p,sigma_r))
                                    else:
                                        contrib_local[v][event[jjj]] = partial_sum[l_nei[v,t][ii+1][0]]  + contribution[w][t_p]*(self.coef_volume_dis(node, v,event[jjj],w,t_p,sigma_r))
                                #print("contrib_local[v][event[jjj]]",contrib_local[v][event[jjj]],"partial_sum[l_ord[ii]]",partial_sum[l_ord[ii]],"contribution[w][t_p]",contribution[w][t_p],"res",res)

            if v not in contribution:
                contribution[v] = dict()
            for vv in contrib_local:
                for ss in contrib_local[vv]:
                    contribution[vv][ss] = contrib_local[vv][ss]
            contribution[v][t] = s + deltasvvt[(v,t)]
        print("******** end call contri_delta_svt","v", v, "t", t,"contribution[v][t]",contribution[v][t])
        return contribution


    #############################################################################
    #                       continuous                                          #
    #############################################################################

    def predecessor_graph_con(self,pre, node):
        G = nx.DiGraph()
        for k in self.nodes:
            if k != node:
                for key in pre[k].keys():
                    for v2 in pre[k][key].keys(): 
                        v,t = v2
                        G.add_edge((v,t),(k,key),interval=pre[k][key][v2])

        return G

    def graph_to_ordered_con(self, G, ev, ev_rev):
        return con.OrdGraphCon(G.nodes, G.edges, ev, ev_rev, self.sinks(G))

    def vol_rec_con_inst(self, s, e, G_rev, sigma):
        l = list(G_rev[e])
        print(e,l,sigma)
        w,tp = l[0]
        sigma[e] = dict()
        if w == s:
            t1,t2 = G_rev[e][(w,tp)]['interval']
            if t1 == t2:
                sigma[e][0] = nppol.Polynomial([1])
                sigma[e][-1] = sigma[e][0]
            else:
                sigma[e][0] = nppol.Polynomial([1])
                sigma[e][1] = nppol.Polynomial([0,(t2-t1)])
                sigma[e][-1] = sigma[e][0] + sigma[e][1]
        else:
            res = 0
            for (w,tp) in l:
                self.vol_rec_con_inst(s, (w,tp), G_rev, sigma)
                if self.actual_degree(sigma[(w,tp)][-1]) == 0:
                    res += sigma[(w,tp)][-1]
                    if 0 not in sigma[e]:
                        sigma[e][0] = sigma[(w,tp)][-1]
                    else:
                        sigma[e][0] += sigma[(w,tp)][-1]

                else:
                    t1,t2 = G_rev[e][(w,tp)]['interval']
                    coef_zero = sigma[(w,tp)][-1].coef[0]
                    if t1 == t2:
                        coef_zero = sigma[(w,tp)][-1].coef[0]
                        if 0 not in sigma[e]:
                            sigma[e][0] = nppol.Polynomial([sigma[(w,tp)].coef[0]])
                        else:
                            sigma[e][0] += nppol.Polynomial([sigma[(w,tp)].coef[0]])

                        res += nppol.Polynomial([coef_zero])
                    else:
                        d = self.actual_degree(sigma[(w,tp)][-1])
                        coef_max = sigma[(w,tp)][-1].coef[d]
                        coef_res = [0 for j in range(d+2)]
                        coef_res[d+1] = coef_max*(t2 - t1)/(d+1)
                        coef_res[0] = sigma[(w,tp)][-1].coef[0]
                        if 0 not in sigma[e]:
                            sigma[e][0] = nppol.Polynomial([coef_res[0]])
                        else:
                            sigma[e][0] += nppol.Polynomial([coef_res[0]])
                        cof_fin = coef_res[:]
                        cof_fin[0] = 0
                        if d+1 not in sigma[e]:
                            sigma[e][d+1] = nppol.Polynomial(cof_fin)
                        else:
                            sigma[e][d+1] += nppol.Polynomial(cof_fin)


                        res += nppol.Polynomial(coef_res)
            sigma[e][-1] = res

    def vol_rec_con(self, s, e, G_rev, sigma, cur_best, mx):
        if e in sigma:
            return
        if e[0] == s:
            sigma[e] = dict()
            sigma[e][-1] = nppol.Polynomial([0])
            return

        if cur_best[e[0]][e[1]][0] == e[1]:
            #instantenous metapaths
            self.vol_rec_con_inst(s, e, G_rev, sigma)
        else:
            sigma[e] = dict()
            for j in range(0,mx+1):
                if j == 0:
                    l = list(G_rev[e])
                    print(e,l,sigma)
                    res = 0
                    for (w,tp) in l:
                        self.vol_rec_con(s, (w,tp), G_rev, sigma, cur_best, mx)
                        if cur_best[w][tp][0] == tp:
                            coef_zero = sigma[(w,tp)][-1].coef[0]
                            res += nppol.Polynomial([coef_zero])
                        else:
                            res += sigma[(w,tp)][-1]
                    sigma[e][0] = res
                elif j == 1:
                    l = list(G_rev[e])
                    print(e,l,sigma)
                    res = 0
                    for (w,tp) in l:
                        t1,t2 = G_rev[e][(w,tp)]['interval']
                        if t1 != t2 and tp <= t1:
                            self.vol_rec_con(s, (w,tp), G_rev, sigma, cur_best, mx)
                            if cur_best[w][tp][0] == tp:
                                coef_zero = sigma[(w,tp)][-1].coef[0]
                                res += nppol.Polynomial([0,coef_zero*(t2 - t1)])
                            else:
                                res += sigma[(w,tp)][-1] * nppol.Polynomial([0,(t2 - t1)])
                    sigma[e][1] = res
                else:
                    l = list(G_rev[e])
                    print(e,l,sigma)
                    res = 0
                    for (w,tp) in l:
                        t1,t2 = G_rev[e][(w,tp)]['interval']
                        if t1 != t2 and tp == t2:
                            self.vol_rec_con(s, (w,tp), G_rev, sigma, cur_best, mx)
                            if (j-1) in sigma[(w,tp)]:
                                res += sigma[(w,tp)][j-1] * nppol.Polynomial([0,(t2 - t1)/j])
                    sigma[e][j] = res
            sigma[e][-1] = sum( sigma[e][jj]  for jj in range(0,mx+1))




    def volume_metapaths_con(self, G, s, cur_best, mx):
        sigma = dict()
        sink = self.sinks(G)
        G_rev = G.reverse(copy=True)
        for e in sink:
            self.vol_rec_con(s, e, G_rev, sigma, cur_best, mx)
        return sigma

    def optimal_with_resting_con(self, node, f_edge, events, G, sigma, cur_best, unt):
        sigma_r = dict()
        for k in self.nodes:
            pred = -1
            for t in events:
                if k == node:
                    sigma_r[(k,t)] = nppol.Polynomial([0])
                else:
                    if pred == -1:
                        if (k,t) in G:
                            sigma_r[(k,t)] = sigma[(k,t)][-1]
                            pred = t
                            edge = f_edge[(k,t)]
                        else:
                            sigma_r[(k,t)] = nppol.Polynomial([0])
                    else:
                        if (k,t) in G:
                            edge2 = f_edge[(k,t)]
                            if edge == edge2 and unt[k][pred] >= t:
                                sigma_r[(k,t)] = sigma_r[(k,pred)] + sigma[(k,t)][-1]
                                pred = t
                            elif edge == edge2 and not(unt[k][pred] >= t):
                                sigma_r[(k,t)] = sigma[(k,t)][-1]
                                pred = t
                            else:
                                #instant paths occur only here
                                if cur_best[k][t][0] == t and self.actual_degree(sigma[(k,t)][-1])>0:
                                    sigma_r[(k,t)] = nppol.Polynomial([sigma[(k,t)][-1].coef[0]])
                                else:
                                    sigma_r[(k,t)] = sigma[(k,t)][-1]
                                pred = t
                                edge = f_edge[(k,t)]
                        else:
                            sigma_r[(k,t)] = sigma_r[(k,pred)]
        return sigma_r

    def check_contri_con(self, j, i,latencies, sigma, node, sigma_p):
        lati = (self.cal_lat_dis(i,latencies),latencies[i][2])
        latj = (self.cal_lat_dis(j,latencies),latencies[j][2])

        if (lati[0] == 0) and lati[1] != 0 and j < i:
            # latencies are instantenous
            print("lat[i]",latencies[i],"latj[j]",latencies[j],"node",node)
            x = -numpy.Infinity
            if (node,latencies[i][0]) in sigma_p:
                x = self.actual_degree(sigma_p[(node,latencies[i][0])][-1])
            if x > 0:
                print("ok = > lat[i]",latencies[i],"latj[j]",latencies[j],"node",node)
                return 2

        if latj < lati:
            return 1
        elif latj == lati:
            x = -numpy.Infinity
            if (node,latencies[j][0]) in sigma_p:
                x = self.actual_degree(sigma_p[(node,latencies[j][0])][-1])
            if x > 0 and j == i+1:
                return 2
            elif x > 0 and not (j == i+1):
                return 1
            else:
                return 0
        else:
            return -1

    def contribution_each_latency_con(self,latencies, mini, maxi, sigma, sigma_p):
        #        contri = [dict() for i in range(len(self.nodes))]
        contri = [dict() for i in range(len(self.nodes))]
        prev_next = [dict() for i in range(len(self.nodes))]
        for k in self.nodes:
            # l contains first arrival times
            #l = [e for e in latencies[k].keys() ]
            #l.sort()
            l = [x for (x,y,z) in latencies[k] ]
            for i in range(0,len(l)):
                #check left contribution
                j = i - 1
                S = -numpy.Infinity
                b = True
                while(j >= 0 and b):
                    #cond = self.check_contri_dis(l[j], l[i], latencies[k], k)
                    cond = self.check_contri_con(j, i, latencies[k], sigma, k, sigma_p)
                    if cond == 2:
                        S = latencies[k][i][1]
                        b = False
                    if  cond == 1:
                        #S = latencies[k][l[j]][0]
                        S = latencies[k][j][1]
                        b = False
                    else:
                        if cond == 0:
                            if l[i] not in prev_next[k]:
                                prev_next[k][l[i]] = [l[j]]
                            else:
                                prev_next[k][l[i]].append(l[j])
                        j = j - 1
                if S == -numpy.Infinity:
                    S = mini
                #check right contribution
                j = i + 1
                A = -numpy.Infinity
                b = True
                while(j < len(l) and b):
                    #cond = self.check_contri_dis(l[j], l[i], latencies[k], k)
                    cond = self.check_contri_con(j, i, latencies[k], sigma, k, sigma_p)
                    if cond == 2:
                        A = l[i]
                        b = False
                    if cond == 1:
                        A = l[j]
                        b = False
                    else:
                        if cond == 0:
                            if l[i] not in prev_next[k]:
                                prev_next[k][l[i]] = [l[j]]
                            else:
                                prev_next[k][l[i]].append(l[j])
                        j = j + 1
                if A == -numpy.Infinity:
                    A = maxi
                contri[k][l[i]] = (S,A)
        return contri,prev_next

    def contri_delta_svvt_con(self, s, v, t, lat, contri, prev_next, sigma_r, lat_rev):
        #if (v,t) in deltasvvt:
            #return deltasvvt[(v,t)]
        print("///////// call svvt, ","s",s,"v",v,"t",t)
        if s == v:
            return 0
        #voir papier matthieu clemence pour l'algo

        if t not in lat[v]:
            return 0
        prev = []
        next = []


        if t in prev_next[v]:
            prev = [lat[v][e][0] for e in prev_next[v][t] if e < t]
            next = [e for e in prev_next[v][t] if e > t]
        #if t not in contri[v]:
        #    t = pointer[(v,t)][1]

        #t_sigma = pointer[(v,t)][1]
        #if (v,t) not in sigma_r:
        #    t_sigma = pointer[(v,t)][1]
        #print("t_contri",v,t)
        #print("t_sigma",v,t_sigma)


        #prev.sort(reverse = True)
        #prev.reverse()
        prev =  prev + [contri[v][t][0]]
        next = next + [contri[v][t][1]]
        #prev.sort()
        print("prev", prev)
        print("next", next)

        left = 0
        right = 0

        contrib = 0
        s_prime = lat[v][t][0]
        vol_tv = sigma_r[(v,t)]
        # print("vol_tv",vol_tv, vol_tv.coef)
        # if self.zero_array(vol_tv.coef):
        #     return nppol.Polynomial([0])
        for s_left in prev:
            # if pointer[(v,s_left)] in sigma_r:
            #     left += sigma_r[pointer[(v,s_left)]][1]
            # else:
            #     left = 0

            a_prime = t
            right = 0
            for a_right in next:
                # if pointer[(v,a_right)] in sigma_r:
                #     right += sigma_r[pointer[(v,a_right)]][1]
                # else:
                #     right = 0

                print("s_prime", s_prime, "s_left", s_left, "a_right", a_right, "a_prime", a_prime)
                enum = (s_prime - s_left) * (a_right - a_prime) * vol_tv
                # print("enum poly", tmp)
                # enum_degree = self.actual_degree(tmp)
                # print("actual enum", enum_degree)
                # enum = tmp
                # if enum_degree == -1:
                #     enum = tuple([0])
                # else:
                #     enum = tuple([tmp.coef[enum_degree] if i == enum_degree else 0 for i in range(enum_degree+1)])



                print("left", left, "vol_tv", vol_tv, "right", right)
                denum = left + vol_tv + right
                # print("denum poly", tmp)
                # denum_degree = self.actual_degree(tmp)
                # print("actual denum", denum_degree)
                # denum = tmp
                #denum = tuple([tmp.coef[denum_degree] if i == denum_degree else 0 for i in range(denum_degree+1)])

                print("enum", enum, "denum", denum)
                #print("enum-denum degree",enum_degree, denum_degree)

                res = self.polydiv(enum,denum)
                if res[1] < 0:
                    ress = [0]
                else:
                    ress =[0 for i in range(res[1]+1)]
                    ress[res[1]] = res[0]
                #contrib += (enum/denum)
                contrib += nppol.Polynomial(ress)
                print("contrib",contrib)
                #if pointer[(v,a_right)] in sigma_r:
                if a_right in lat[v]:
                    right += sigma_r[(v,a_right)]
                # else:
                #     right = 0
                a_prime = a_right
            #if pointer[(v,lat_rev[v][s_left])] in sigma_r:
            if (s_left in lat_rev[v]) and (lat_rev[v][s_left] in lat[v]):
            #if lat_rev[v][s_left] in lat[v]:
                left += sigma_r[(v,lat_rev[v][s_left])]
            # else:
            #     left = 0
            s_prime = s_left
        print("end svvt", contrib)
        #deltasvvt[(v,t)] = contrib
        return contrib

    def coef_volume_con(self, x, v, t, w, t_p, sigma_r, pre, st1t2):
        print("divison_volume","v",v,"t",t,"w",w,"t_p",t_p, "st1t2", st1t2)
        # (t1,t2) = pre[w][t_p][v,t]
        # if v == x :
        #     return nppol.Polynomial([1])
        # if t1 < t:
        #     (t1,t2) = (t2,t2)
        # st1t2 = nppol.Polynomial([1])
        # if t2 > t1:
        #     rr = [0 for ii in range(chev+1)]
        #     rr[chev] = math.pow((t2-t1),chev)/math.factorial(chev)
        #     st1t2 = nppol.Polynomial(rr)
        # print("st1t2",st1t2)
        if x == v:
            return nppol.Polynomial([1])
        svt = sigma_r[(v,t)]
        swtp = sigma_r[(w,t_p)]

        print("svt", svt)
        tmp = self.polymul(svt, st1t2)
        #tmp = tmp[0]
        print("svt*st1t2", tmp)
        svt_high = [0 for jj in range(tmp[1]+1)]
        svt_high[tmp[1]] = tmp[0]
        svt_high = nppol.Polynomial(svt_high)
        svt_degree = self.actual_degree(svt_high)
        print("actual svt*st1t2", svt_degree)
        
        #svt_high = tuple([tmp.coef[svt_degree] if i == svt_degree else 0 for i in range(svt_degree+1)])
        #if svt_high == ():
        #    svt_high = (0)
        print("svt_hight",svt_high)
        print("swtp", swtp)
        tmp = swtp
        swtp_degree = self.actual_degree(tmp)
        print("actual swtp", swtp_degree)
        swtp_high = tmp
        #swtp_high = tuple([tmp.coef[swtp_degree] if i == swtp_degree else 0 for i in range(swtp_degree+1)])
        print("swtp_high", swtp_high)
        if swtp_degree != -1:
            cof,puis = self.polydiv(svt_high,swtp_high)
            if puis < 0:
                res = [0]
            else:
                res = [0 for i in range(puis+1)]
                res[puis] = cof
        else:
            res = [0]
        print("res_div", res)
        return nppol.Polynomial(res)


    def contri_delta_svt_con(self, node, v, t, l_nei, sigma_r, contribution, deltasvvt, event, event_reverse, pre, GT, unt):
        print("******** new call contri_delta_svt","v", v, "t", t)
        if (v not in contribution) or ((v in contribution) and (t not in contribution[v])):
            partial_sum = dict()
            s = 0
            contrib_local = dict()
            # l_ord = list(dic_nodes_rev.keys())
            # l_ord.sort()
            for ii in range(len(l_nei[(v,t)])-1,-1,-1):
                #normally it should be sorted
                for u in l_nei[v,t][ii][1]:
                    w,t_p = (u,l_nei[v,t][ii][0])
                    print("(w,t')",(w,t_p))
                    self.contri_delta_svt_con(node, w, t_p, l_nei, sigma_r, contribution, deltasvvt, event, event_reverse, pre, GT)
                    (t1,t2) = pre[w][t_p][v,t]
                    if t1 != t2 and t_p > t and unt(t) >=tp:
                        for yp,tpp in GT[t1,t2][w,t_p]:
                            print("*!*!*!*!*!*!  instant graph","v",v,"t",t,"t1",t1,"t2",t2, "w",w,"t_p",t_p, "yp", yp, "tpp", tpp)
                            self.contri_delta_svt_con(node, yp, tpp, l_nei, sigma_r, contribution, deltasvvt, event, event_reverse, pre, GT)
                            chev = GT[t1,t2][w,t_p][yp,tpp]["weight"]+1
                            rr = [0 for ii in range(chev+1)]
                            rr[chev] = math.pow((t2-t1),chev)/math.factorial(chev)
                            res2 = self.coef_volume_con(node, v, t, yp, tpp, sigma_r, pre, nppol.Polynomial(rr))

                            #res2 *= nppol.Polynomial(rr)
                            s += res2 * contribution[yp][tpp]
                    if t_p > t and t2 > t1:
                        res = self.coef_volume_con(node, v, t, w, t_p, sigma_r, pre, nppol.Polynomial([0,(t2-t1)]))
                    else:
                        res = self.coef_volume_con(node, v, t, w, t_p, sigma_r, pre, nppol.Polynomial([1]))

                    
                    #appel recursif
                    s += res * contribution[w][t_p]
                    if l_nei[v,t][ii][0] not in partial_sum:
                        partial_sum[l_nei[v,t][ii][0]] = s
                    else:
                        partial_sum[l_nei[v,t][ii][0]] += s

                    print("******** half call contri_delta_svt","v", v, "t", t, "sum", s)
                    if ii != 0:
                        jj = event_reverse[l_nei[v,t][ii-1][0]]
                    else:
                        jj = event_reverse[t]
                    print("u", u, "dic_nodes[u]", l_nei[v,t][ii])
                    print("******** half after call contri_delta_svt","v", v, "t", t)
                    print("dic_nodes[u]",l_nei[v,t],"u",u)
                    if True:#ii != len(dic_nodes[u])-1:
                        for jjj in range(jj+1,event_reverse[l_nei[v,t][ii][0]]+1):
                            print("v", v, "t",t,"event[jj]", event[jj], "dic_nodes[u][ii]", "index actual event",jj,"index succ events", jjj, "contri event time" ,event[jjj])
                            print("comp vol", sigma_r[v,t], sigma_r[v,event[jjj]])

                            if v not in contrib_local:
                                contrib_local[v] = dict()
                            if not (v in contribution and event[jjj] in contribution[v]):
                                if sigma_r[v,t] != sigma_r[v,event[jjj]]:
                                    print("ERREUUUUUUUR ",sigma_r[v,t],sigma_r[v,event[jjj]])
                                print("add_contri_local","v",v,"event[jjj]",event[jjj],"w,t_p",w,t_p)
                                if event[jjj] != t_p:
                                    print("ici")
                                    contrib_local[v][event[jjj]] = partial_sum[l_nei[v,t][ii][0]]
                                else:
                                    print("la")
                                    if ii == len(l_nei[v,t])-1:
                                        contrib_local[v][event[jjj]] =  contribution[w][t_p]*(self.coef_volume_con(node, v,event[jjj],w,t_p,sigma_r, pre, nppol.Polynomial([1]) ))
                                    else:
                                        contrib_local[v][event[jjj]] = partial_sum[l_nei[v,t][ii+1][0]]  + contribution[w][t_p]*(self.coef_volume_con(node, v,event[jjj],w,t_p,sigma_r, pre, nppol.Polynomial([1])))
                                #print("contrib_local[v][event[jjj]]",contrib_local[v][event[jjj]],"partial_sum[l_ord[ii]]",partial_sum[l_ord[ii]],"contribution[w][t_p]",contribution[w][t_p],"res",res)

            if v not in contribution:
                contribution[v] = dict()
            for vv in contrib_local:
                for ss in contrib_local[vv]:
                    contribution[vv][ss] = contrib_local[vv][ss]
            contribution[v][t] = s + deltasvvt[(v,t)]
        print("******** end call contri_delta_svt","v", v, "t", t,"contribution[v][t]",contribution[v][t])
        return contribution







    #############################################################################
    #                       END                                                 #
    #############################################################################


def postprocess_kcliques_into_clusters(Kcliques):
    """
    We only keep the biggest 'K' for each segmented temporal nodes

    :param Kcliques:
    :return:
    """
    K = defaultdict(list)
    seen = set()
    for k, v in sorted(Kcliques.items(), reverse=True):
        for clique in v:
            for e in clique:
                if e not in seen:
                    seen.add(e)
                    K[k].append(e)
    return K


def adjacency_list_to_json(a_l):
    js = {'nodes': [],
          'links': []}
    for u in a_l:
        js['nodes'].append({'id': u})
        for v in a_l[u]:
            js['links'].append({'source': u,
                                'target': v})
    return js


def write_adjacency_list_to_json(a_l, path_json):
    js = {'nodes': [],
          'links': []}
    for u in a_l:
        js['nodes'].append({'id': u})
        for v in a_l[u]:
            js['links'].append({'source': u,
                                'target': v})
    with open(path_json, 'w') as file_output:
        json.dump(js, file_output)


def get_index_node_to_id_wcc(list_WCC):
    """
    For source node paths

    :param list_WCC:
    :return:
    """
    index_node_to_wcc = defaultdict(set)
    for SG in list_WCC:
        for n in SG.nodes:
            index_node_to_wcc[n].add(SG.id)
    return index_node_to_wcc


def get_index_segmented_node_to_id_wcc(list_WCC):
    """
    for temporal source node paths

    :param list_WCC:
    :return:
    """
    index_segmented_nodes_to_wcc = {}
    for SG in list_WCC:
        for n, np in zip(SG.nodes, SG.node_presence):
            for t0, t1 in zip(np[::2], np[1::2]):
                index_segmented_nodes_to_wcc[(t0, t1, n)] = [SG.id]
    return index_segmented_nodes_to_wcc


def condensation_dag_from_wcc(list_WCC):
    """
    Get the corresponding condensation DAG from a list of Weakly Connected Components (Stream Graphs Objects)

    :param list_WCC:
    :return:
    """
    n_wcc = len(list_WCC)

    def para_cdag(SG):
        return SG.CondensationDag()

    if n_wcc >= 4:
        # We parallelize only if the number of WCC is high enough :)
        list_CDAG = Parallel(n_jobs=-1, mmap_mode='r+')(
            delayed(para_cdag)(SG) for SG in list_WCC)
    else:
        list_CDAG = [SG.CondensationDag() for SG in list_WCC]
    return list_CDAG


def test_all_fastest_paths_dag(S):
    new_node_presence = [[S.times[0], S.times[1]] for _ in S.nodes]
    S.node_presence = new_node_presence
    print("Nb of WCC :", S.number_of_weakly_connected_component())
    dag = S.CondensationDag()
    dag.describe()
    dag.plot()
    test_node = None
    for n, npres in zip(S.nodes, S.node_presence):
        for t0, t1 in zip(npres[::2], npres[1::2]):
            source = (t0, t1, n)
            if test_node is None:
                test_node = source
            if source[2] != test_node[2]:
                print("\nSource :", source)
                print("Destination :", test_node)
                fp, lat1 = dag.fastest_path(source, test_node)
                all_fp, lat2 = dag.all_fastest_paths(source, test_node)
                assert lat1 == lat2
                if fp:
                    # print("fp :", fp)
                    # print("all fp :", all_fp)
                    for p in all_fp:
                        sub = dag.path_induced_substream(p)
                        sub.plot()
                        # dag.plot_fastest_path(fp, lat2, source, destination=test_node)
    plt.show()


def performance_test_path_functions(S, path_type, random_nodes=False,
                                    high_degree_nodes=False):
    dag = S.CondensationDag()
    path_functions = {'latencies': S.latencies,
                      'distances': S.distances,
                      'ttr': S.times_to_reach,
                      'FSP': S.fastest_shortest_path,
                      'SFP': S.shortest_fastest_path,
                      'dag_ttr': dag.times_to_reach,
                      'dag_latencies': dag.latencies}

    # TODO : Rajouter un dataframe pour les résultats
    time_begin = time.time()
    if random_nodes:
        source_nodes = numpy.random.choice(S.nodes, 10)
        source_nodes = [(t0, t1, n) for n, npres in zip(S.nodes, S.node_presence)
                        for t0, t1 in zip(npres[::2], npres[1::2]) if n in source_nodes]
    elif high_degree_nodes:
        d = S.degrees_partition()
        source_nodes = []
        for degree, n in sorted(d.items(), reverse=True):
            for i in n:
                source_nodes.append(i[2])
                if len(source_nodes) == 10:
                    break
            if len(source_nodes) == 10:
                break
        source_nodes = [(t0, t1, n) for n, npres in zip(S.nodes, S.node_presence)
                        for t0, t1 in zip(npres[::2], npres[1::2]) if n in source_nodes]
    else:
        source_nodes = [(t0, t1, n) for n, npres in zip(S.nodes, S.node_presence)
                        for t0, t1 in zip(npres[::2], npres[1::2])]
    nb_paths_to_compute = len(source_nodes) * (len(S.nodes) - 1)
    print("Number of paths to compute :", nb_paths_to_compute)
    cnt_null_latencies = 0
    cnt_non_null_latencies = 0
    cnt_none_latencies = 0
    for u in source_nodes:
        R = path_functions[path_type](u)
        for v in S.nodes:
            if v != u:
                if v not in R:
                    cnt_none_latencies += 1
                    # print("No paths between ", u, " and ", v, " !")
                elif R[v] == 0:
                    cnt_null_latencies += 1
                    # print("Latency from ", u, " to ", v, " :", L[v])
                else:
                    cnt_non_null_latencies += 1
                    # print("Latency from ", u, " to ", v, " :", L[v])
    time_end = time.time() - time_begin
    print("Paths latencies computed in ", time_end)
    print("Time per latency :", time_end / nb_paths_to_compute)
    print("None latencies :", cnt_none_latencies)
    print("Null latencies :", cnt_null_latencies)
    print("Strictly Positive latencies :", cnt_non_null_latencies)

