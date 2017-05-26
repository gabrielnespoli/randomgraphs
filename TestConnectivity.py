import sys
import numpy as np
import time
from numpy import linalg as LA
randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')


#function needed for bfs
def matrix_to_dict(matrix):
    graph = {}
    for i, node in enumerate(matrix):
        # adj = []
        # for j, connected in enumerate(node):
        #     if connected:
        #         adj.append(j)
        # graph[i] = adj
        graph[i] = list(np.where(matrix[i] != 0)[0])
    return graph


def bfs(graph, v):
    all = []
    Q = []
    Q.append(v)
    while Q != []:
        v = Q.pop(0)
        all.append(v)
        for n in graph[v]:
            if n not in Q and n not in all:
                Q.append(n)
    return all


def is_connected_method1(L):
    is_connected = False
    k = np.shape(L)[0]  # take the dimesion of Laplacian matrix
    P = np.identity(k)
    A = np.diag(np.diag(L)) - L
    sum_B = P
    for i in range(1, k):
        sum_B += LA.matrix_power(A, i)  # I+A+A^2....>0
    s = sum_B
    if s.all() > 0:
        is_connected = True
    return is_connected


def is_connected_method2(rand_graph):
    is_connected = False
    eigenValues = LA.eig(rand_graph)[0]
    secmin = sorted(eigenValues)[1]  ### sort and pick second
    if (secmin > 0.001):
        is_connected = True
    return is_connected


def is_connected_method3(rand_graph, n, start):
    is_connected = False
    lst = matrix_to_dict(rand_graph)
    breads = bfs(lst, start)
    if n == len(breads):
        is_connected = True
    return is_connected


def mean_complex(sim, nodes, prob, n_graph, n):
    v1 = []
    v2 = []
    v3 = []

    if (n == 1):
        for i in range(0, n_graph):
            g = randomGraph.buildRandomGraph(nodes, prob)
            for j in range(sim):  # calculate several times complexity on the same graph
                t1 = time.time()  # to be more precisely
                m1 = is_connected_method1(g)
                v1.append(time.time() - t1)

        print(v1)
        print("mean of is_connected_method1:", np.mean(v1))

    if (n == 2):
        for i in range(0, n_graph):
            g = randomGraph.buildRandomGraph(nodes, prob)

            for k in range(sim):
                t2 = time.time()
                m2 = is_connected_method2(g)
                v2.append(time.time() - t2)
        print(v2)
        print("mean of is_connected_method2:", np.mean(v2))

    if (n == 3):
        for i in range(0, n_graph):
            g = randomGraph.buildRandomGraph(nodes, prob)

            for l in range(sim):
                t3 = time.time()
                m3 = is_connected_method3(g, nodes, 1)
                v3.append(time.time() - t3)

        print(v3)
        print("mean of is_connected_method3:", np.mean(v3))


def main():
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    p = float(sys.argv[3])
    startingNode =0

    rand_graph = randomGraph.buildRandomGraph(n, p)
    print(rand_graph)

    print("\n")
    if is_connected_method1(rand_graph):
        print("Connected.M1")

    if is_connected_method2(rand_graph):
        print("Connected.M2")

    if is_connected_method3(rand_graph, n, startingNode):
        print("Connected.M3")

    rand_reg_graph = regRandGraph.RegularGraph(n, r, p)
    a = rand_reg_graph.build_regular_graph()
    print(a)
    print("M1 ", is_connected_method1(a),"\nM2 ", is_connected_method2(a),"\nM3 ", is_connected_method3(a, n, startingNode))

    v1 = []
    v2 = []
    v3 = []

    sim = 10

    for i in range(0, 3):
        g = randomGraph.buildRandomGraph(100, 80)

        for j in range(sim):
            t1 = time.time()
            m1 = is_connected_method1(g)
            v1.append(time.time() - t1)

        for k in range(sim):
            t2 = time.time()
            m2 = is_connected_method2(g)
            v2.append(time.time() - t2)

        for l in range(sim):
            t3 = time.time()
            m3 = is_connected_method3(g, 8, 3)
            v3.append(time.time() - t3)

    print(v1)
    print(v2)
    print(v3)

    mean_m1, mean_m2, mean_m3 = np.mean(v1), np.mean(v2), np.mean(v3)
    print("mean method 1:", mean_m1, "mean method 2:", mean_m2, "mean method 3", mean_m3)

if __name__ == "__main__":
    main()

