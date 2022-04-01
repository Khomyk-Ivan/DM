import copy
from collections import defaultdict
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.ROW = len(graph)

    def BFS(self, s, t, parent):
        visited = [False] * self.ROW

        queue = [s]

        visited[s] = True

        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True

        return False

    def FordFulkerson(self, source, sink):

        parent = [-1] * self.ROW

        max_flow = 0

        while self.BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            max_flow += path_flow

            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow


def getFile():
    global text
    text = []
    text_str = []
    with open("l4-2.txt", "r") as file:
        lines = file.readlines()
        [text_str.append(i.replace('\n', '')) for i in lines]
    for i in range(len(text_str)):
        text_str[i] = text_str[i].split(" ")
    for i in text_str:
        list = []
        for k in i:
            list.append(int(k))
        text.append(list)
    text.pop(0)


def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()

def create_dict(graph_matrix):
    new_dict = {}
    for i in range(len(graph_matrix)):
        new_dict[i] = i + 1

    return new_dict


def data_for_graph(matrix):
    matrix = np.array(matrix)
    matrix[matrix > 1] = 1

    matrix_dict = create_dict(matrix)
    show_graph_with_labels(matrix, matrix_dict)


def start_counting():
    global answer
    g = Graph(text)
    answer = g.FordFulkerson(source-1, sink-1)
    return answer


if __name__ == '__main__':
    getFile()

    source = int(input(f"Оголосити вихідну вершину в діапазоні 1 - {len(text)}: "))
    sink = int(input(f"Оголосити вершину приймача 1 - {len(text)}: "))

    start_counting()

    if answer == 0:
        data_for_graph(text)
        print("\nПотоку немає, довжина струму є, ", answer)
        a= copy.deepcopy(sink)-1
        while answer == 0:
            if text[a-1][a] == 0:
                text[a-1][a] = randint(10, 40)
            else:
                a -= 1
            start_counting()

        data_for_graph(text)
        print("\nМаксимально можливий потік ", answer)

    else:
        data_for_graph(text)

        print("\nМаксимально можливий потік ", answer)

        for i in range(len(text)):
            if text[i][sink-1] > 0:
                if text[sink-1][i] > 0:
                    text[i][sink-1] = 0
                else:
                    text[sink-1][i] = text[i][sink-1]
                    text[i][sink-1] = 0

        start_counting()

        data_for_graph(text)
        print("\nПотоку немає, довжина струму є ", answer)


