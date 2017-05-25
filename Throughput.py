import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')
dp = __import__('ShortestDisjointPaths')


def throughput(origG, l):
    n = len(origG)
    links = np.zeros((n, n,))

    # we use the fact that the graph is undirected
    for i in range(n - 1):
        j = i + 1
        while j < n:
            matrix = np.copy(origG)
            sel = dp.Graph(matrix)
            disj_path = sel.find_disjoint_paths(i, j, l)
            paths = disj_path[1]
            true_l = disj_path[0] #True number of disjoint path
            if true_l != 0:
                for path in paths:
                    k = 0
                    while k < len(path) - 1:
                        if (path[k] < path[k + 1]):
                            links[path[k], path[k + 1]] = links[path[k], path[k + 1]] + 1/true_l
                        else:
                            links[path[k + 1], path[k]] = links[path[k + 1], path[k]] + 1/true_l
                        k = k + 1

            j = j + 1
    m = links.max()
    thr = 1 / m
    return thr

def destroyLink(matrix, i, j):
    matrix[i, j] = 0
    matrix[j, i] = 0
    matrix[i, i] = matrix[i, i] - 1
    matrix[j, j] = matrix[j, j] - 1

def linkFail (mat, p, l, n_fail):
    matrix = np.copy(mat)
    failure = 0
    if n_fail != 0:
        for i in range(len(mat)):
            for j in range(len(mat)):
                if (matrix[i, j] == -1):
                    k = np.random.choice([0, 1], p = [1 - p, p])
                    if (k == 1):
                        destroyLink(matrix, i, j)
                        failure = failure + 1
                        if(n_fail == failure):
                            break

    thr = throughput(matrix, l)
    return thr

#plot of the Throughput with fix p and variables n and l
def plotThroughpt (p):
    tot_l = []
    l = [1,2,3,4]
    n = list(np.arange(10, 110, 10))
    for i in l:
        thr = []
        for j in n:
            mat = randomGraph.buildRandomGraph(j, p)
            s = throughput(mat, i)
            thr.append(s)
        tot_l.append(thr)

    plt.plot(n, tot_l[0], label="l = 1")
    plt.plot(n, tot_l[1], label="l = 2")
    plt.plot(n, tot_l[2], label="l = 3")
    plt.plot(n, tot_l[3], label="l = 4")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)

    plt.xlabel('Nodes')
    plt.ylabel('Throughput')
    plt.axis([10, 100, 0, 1])
    plt.show()

#plot throughput with a fix failure

def plotThroughputFail( n, p, prob, n_fail):
    mat = randomGraph.buildRandomGraph(n, p)
    list = []
    l = [1, 3]
    for i in l:
        thr = []
        for p in prob:
            t = linkFail(mat, p, i, n_fail)
            thr.append(t)
        list.append(thr)

    plt.plot(prob, list[0], 'r', label="l = 1")
    plt.plot(prob, list[1], 'g', label="l = 3")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.xlabel('Probabilities')
    plt.ylabel('Throughput')
    plt.axis([0.0, 0.25, 0, 1])
    plt.show()



def main():

    prob = np.arange(0.01, 0.25, 0.01)
    p = 0.6

    plotThroughpt(p)
    plotThroughputFail(10, p, prob,4)



if __name__ == "__main__":
    main()