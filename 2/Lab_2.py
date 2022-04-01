import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from random import randint

def getFile():
    global agency_matrix
    agency_matrix = []; text_str = []
    with open("1.txt", "r") as file:
        lines = file.readlines()
        [text_str.append(i.replace('\n', '')) for i in lines]
    for i in range(len(text_str)):
        text_str[i] = text_str[i].split(" ")
    for i in text_str:
        list = []
        for k in i:
            list.append(int(k))
        agency_matrix.append(list)
    agency_matrix.pop(0)

def check_matrix(matrix):
    list_to_check = []
    for rows in range(len(matrix)):
        eq = 0
        for columns in range(len(matrix[0])):
            if matrix[rows][columns] > 0:
                eq += 1
        list_to_check.append(eq)

    return list_to_check


def recreate_graph_with_Eulerian_path(matrix, list_to_change):
    for i in list_to_change:
        num_zero = np.count_nonzero(matrix[i])
        if num_zero > 1:
            for k in list_to_change:
                if k == i:
                    continue
                elif matrix[i][k] != 0:
                    if np.count_nonzero(matrix[k]) > 2:
                        matrix[i][k] = 0
                        matrix[k][i] = 0
                        list_to_change.pop(list_to_change.index(i))
                        list_to_change.pop(list_to_change.index(k))
                        break
                    else:
                        continue
                else:
                    continue

    if len(list_to_change) == 2:
        if matrix[list_to_change[0]][list_to_change[1]] == 0:
            matrix[list_to_change[0]][list_to_change[1]] = 1
            matrix[list_to_change[1]][list_to_change[0]] = 1


def recreate_graph_without_Eulerian_path(matrix):
    for k in range(0,1):
        for i in range(len(matrix[k])):
            if matrix[k][i] > 0:
                matrix[k][i] = 0
                matrix[i][k] = 0
                break


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


def sub(visited, _cur, graph):
    if not graph:
        return visited + [_cur]
    for i, edge in enumerate(graph):
        cur, nex = edge
        if _cur not in edge:
            continue
        _graph = graph[:]
        del _graph[i]
        if _cur == cur:
            res = sub(visited + [cur], nex, _graph)
        else:
            res = sub(visited + [nex], cur, _graph)
        if res:
            return res


def find_eulerian_tour(graph):
    head, tail = graph[0], graph[1:]
    prev, nex = head
    return sub([prev], nex, tail)


def count_Euler(matrix):
    matrix = np.array(matrix)
    matrix[matrix > 1] = 1

    matrix_dict = create_dict(matrix)
    show_graph_with_labels(matrix, matrix_dict)

    list_of_tuples = []
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            if matrix[i][k] > 0:
                list_of_tuples.append((i+1, k+1))
                matrix[i][k] = 0
                matrix[k][i] = 0

    result = find_eulerian_tour(list_of_tuples)

    if result:
        if result[0] != result[-1]:
            print("Цей графік не має ейлерівської схеми, але має тур Ейлера.")
            print("Шлях: ", result, "\n")

        else:
            print("Шлях: ", result)

            length_of_circuit = 0
            for i in range(len(result)-1):
                if agency_matrix[result[i] - 1][result[i + 1] - 1] == 0:
                    agency_matrix[result[i] - 1][result[i + 1] - 1] = randint(20, 60)
                    agency_matrix[result[i + 1] - 1][result[i] - 1] = agency_matrix[result[i] - 1][result[i + 1] - 1]
                    length_of_circuit += agency_matrix[result[i] - 1][result[i + 1] - 1]
                else:
                    length_of_circuit += agency_matrix[result[i] - 1][result[i + 1] - 1]
            print("Вага графа  = ", length_of_circuit, '\n')
    else:
        print("Граф не має Ейлерів цикл\n")


if __name__  == '__main__':
    getFile()

    matrix1 = copy.deepcopy(agency_matrix)
    matrix2 = copy.deepcopy(agency_matrix)

    list_to_check = check_matrix(matrix1)

    list_to_change = []

    for i in range(len(list_to_check)):
        if list_to_check[i] % 2 != 0:
            list_to_change.append(i)

    if list_to_change:
        count_Euler(matrix1)

        recreate_graph_with_Eulerian_path(matrix2, list_to_change)
        count_Euler(matrix2)

    else:
        count_Euler(matrix1)
        recreate_graph_without_Eulerian_path(matrix2)

        count_Euler(matrix2)
