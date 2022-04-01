import math
from random import randint
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
maxsize = float('inf')


def copyToFinal(curr_path):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]

def firstMin(adj, i):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k]

    return min

def secondMin(adj, i):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]

        elif(adj[i][j] <= second and
             adj[i][j] != first):
            second = adj[i][j]

    return second

def TSPRec(adj, curr_bound, curr_weight,
              level, curr_path, visited):
    global final_res

    if level == N:

        if adj[curr_path[level - 1]][curr_path[0]] != 0:

            curr_res = curr_weight + adj[curr_path[level - 1]]\
                                        [curr_path[0]]
            if curr_res < final_res:
                copyToFinal(curr_path)
                final_res = curr_res
        return

    for i in range(N):

        if (adj[curr_path[level-1]][i] != 0 and
                            visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]

            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) +
                                firstMin(adj, i)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) +
                                 firstMin(adj, i)) / 2)

            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True

                TSPRec(adj, curr_bound, curr_weight,
                       level + 1, curr_path, visited)

            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp

            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True

def TSP(adj):
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N

    for i in range(N):
        curr_bound += (firstMin(adj, i) +
                       secondMin(adj, i))

    curr_bound = math.ceil(curr_bound / 2)

    visited[0] = True
    curr_path[0] = 0

    TSPRec(adj, curr_bound, 0, 1, curr_path, visited)


def getFile():
    global text
    text = []; text_str = []
    with open("l3-3.txt", "r") as file:
        lines = file.readlines()
        [text_str.append(i.replace('\n', '')) for i in lines]
    for i in range(len(text_str)):
        text_str[i] = text_str[i].split(" ")
    for i in text_str:
        list = []
        for k in i:
            list.append(int(k))
        text.append(list)


def check_matrix(matrix):
    list_to_check = []
    for rows in range(len(matrix)):
        eq = 0
        for columns in range(len(matrix[0])):
            if matrix[rows][columns] == 0:
                eq += 1
        list_to_check.append(eq)
    return list_to_check


def counting(matrix, num_of_rows):
    global adj, N, final_res, final_path, visited
    adj = matrix
    N = num_of_rows

    final_path = [None] * (N + 1)

    visited = [False] * N

    final_res = maxsize

    TSP(adj)

    print("Minimum cost :", final_res)
    print("Path Taken : ", end = ' ')
    try:
        for i in range(N + 1):
            print(final_path[i]+1, end = ' ')
    except:
        print("The graph doesn't have hamilton's cycle")

    return final_res


def create_a_complete_graph():
    list_to_check = check_matrix(text)

    complete_graph = copy.deepcopy(text)
    for i in range(len(list_to_check)):
        if list_to_check[i] > 4:
            k = 0
            while complete_graph[i].count(0) > 4:
                if complete_graph[i][k] > 0 or i == k:
                    k += 1
                else:
                    rand_number = randint(20, 80)
                    complete_graph[i][k] = rand_number
                    complete_graph[k][i] = rand_number
                    k += 1
    return complete_graph


def create_an_incomplete_graph():
    list_to_check = check_matrix(text)

    incomplete_graph = copy.deepcopy(text)
    posit = list_to_check.index(min(list_to_check))

    k = 0
    while incomplete_graph[posit].count(0) < 5:
        if incomplete_graph[posit][k] == 0:
            k += 1
        else:
            incomplete_graph[posit][k] = 0
            incomplete_graph[k][posit] = 0
            k += 1
    return incomplete_graph


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


if __name__ == "__main__":
    getFile()

    num_of_rows = text[0][0]
    text.pop(0)

    matrix_connections = np.array(text)
    matrix_connections[matrix_connections > 1] = 1
    matrix_dict = create_dict(matrix_connections)
    # print(matrix_connections)

    show_graph_with_labels(matrix_connections, matrix_dict)

    final_res = counting(text, num_of_rows)

    print()
    if final_res == float('inf'):
        new_graph = create_a_complete_graph()
    else:
        new_graph = create_an_incomplete_graph()

    matrix_connections = np.array(new_graph)
    matrix_connections[matrix_connections > 1] = 1
    matrix_dict = create_dict(matrix_connections)

    show_graph_with_labels(matrix_connections, matrix_dict)

    final_res = counting(new_graph, num_of_rows)
