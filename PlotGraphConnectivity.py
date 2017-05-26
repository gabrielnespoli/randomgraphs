import numpy as np
import matplotlib.pyplot as plt
randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')
import TestConnectivity as tc


def plot_prob_connectivity(run=40):
    # print("Calculating the probability of connectivity of the Erdos-Renyi graph...")
    # fig, all_plots = plt.subplots(3, 2)
    # fig.suptitle("Connectivity probability of random graphs", fontsize=16)
    #
    # # generate the Y and X axis for the connectivity probability graph for the ER-Graph
    # y_values = []
    # prob_values = np.arange(0, 1, 0.005)
    # for p in prob_values:
    #     number_connected = 0
    #     for i in range(run):
    #         g_er = randomGraph.buildRandomGraph(n=100, p=p)
    #         number_connected += tc.is_connected_method2(g_er)
    #     y_values.append(number_connected/run)
    #     print("graph probability p =", p, "--> Connectivity probability =", number_connected/run)
    #
    #     j = len(y_values) - 1
    #     if j >= 2:
    #         if y_values[j] != 0 and y_values[j] == y_values[j - 3] and y_values[j] == y_values[j - 2] \
    #                 and y_values[j] == y_values[j - 1]:
    #             break
    #
    # plt.plot(prob_values[0:j+1], y_values)
    # plt.title('Erdos-Renyi graph, n=100')
    # plt.ylabel('Connectivity probability')
    # plt.xlabel('Probability p to draw an arc')
    # plt.show(block=True)
    # all_plots[0, 0].plot(prob_values[0:j+1], y_values)
    # all_plots[0, 0].set_title('Erdos-Renyi graph, n=100')
    #
    # print("Calculating the probability of connectivity of the 2-Regular graph...")
    # # generate the Y and X axis for the connectivity probability graph for the R-Regular-Graph
    # comb_rn_values = [(2, 5), (2, 10), (2, 15), (2, 20), (2, 25), (2, 30), (2, 35), (2, 40), (2, 45), (2, 50)]
    # x_values = []
    # y_values = []
    # for r, n in comb_rn_values:
    #     number_connected = 0
    #     for i in range(run):
    #         RG = regRandGraph.RegularGraph(n=n, r=r, p=0.1)
    #         g_reg = RG.build_regular_graph()
    #         number_connected += tc.is_connected_method2(g_reg)
    #     print("n =", n, ", r =", r, "--> Connectivity probability =", number_connected / run)
    #     x_values.append(n)
    #     y_values.append(number_connected / run)
    # plt.plot(x_values, y_values)
    # plt.title('r = 2')
    # plt.ylabel('Connectivity probability')
    # plt.xlabel('Number of nodes')
    # plt.show(block=True)
    #     # all_plots[1, 0].plot(x_values, y_values)
    #     # all_plots[1, 0].set_title('r = 2')
    #
    # print("Calculating the probability of connectivity of the 4-Regular graph...")
    # # generate the Y and X axis for the connectivity probability graph for the R-Regular-Graph
    # comb_rn_values = [(4, 20), (4, 60), (4, 100)]
    # x_values = []
    # y_values = []
    # for r, n in comb_rn_values:
    #     number_connected = 0
    #     for i in range(run):
    #         RG = regRandGraph.RegularGraph(n=n, r=r, p=0.1)
    #         g_reg = RG.build_regular_graph()
    #         number_connected += tc.is_connected_method2(g_reg)
    #     print("n =", n, ", r =", r, "--> Connectivity probability =", number_connected / run)
    #     x_values.append(n)
    #     y_values.append(number_connected / run)
    # plt.plot(x_values, y_values)
    # plt.title('r = 4')
    # plt.ylabel('Connectivity probability')
    # plt.xlabel('Number of nodes')
    # plt.show(block=True)
    #     # all_plots[1, 1].plot(x_values, y_values)
    #     # all_plots[1, 1].set_title('r = 4')

    print("Calculating the probability of connectivity of the 8-Regular graph...")
    # generate the Y and X axis for the connectivity probability graph for the R-Regular-Graph
    comb_rn_values = [(8, 40), (8, 70), (8, 100)]
    x_values = []
    y_values = []
    for r, n in comb_rn_values:
        number_connected = 0
        for i in range(run):
            RG = regRandGraph.RegularGraph(n=n, r=r, p=0.1)
            g_reg = RG.build_regular_graph()
            number_connected += tc.is_connected_method2(g_reg)
        print("n =", n, ", r =", r, "--> Connectivity probability =", number_connected / run)
        x_values.append(n)
        y_values.append(number_connected / run)
    plt.plot(x_values, y_values)
    plt.title('r = 8')
    plt.ylabel('Connectivity probability')
    plt.xlabel('Number of nodes')
    plt.show(block=True)
    # all_plots[2, 0].plot(x_values, y_values)
    # all_plots[2, 0].set_title('r = 8')

    print("Calculating the probability of connectivity of the 16-Regular graph...")
    # generate the Y and X axis for the connectivity probability graph for the R-Regular-Graph
    comb_rn_values = [(16, 40), (16, 70), (16, 100)]
    x_values = []
    y_values = []
    for r, n in comb_rn_values:
        number_connected = 0
        for i in range(run):
            RG = regRandGraph.RegularGraph(n=n, r=r, p=0.1)
            g_reg = RG.build_regular_graph()
            number_connected += tc.is_connected_method2(g_reg)
        print("n =", n, ", r =", r, "--> Connectivity probability =", number_connected / run)
        x_values.append(n)
        y_values.append(number_connected / run)
    # all_plots[2, 1].plot(x_values, y_values)
    # all_plots[2, 1].set_title('r = 16')

    plt.show(block=True)


def main():
    plot_prob_connectivity()



if __name__ == "__main__":
    main()