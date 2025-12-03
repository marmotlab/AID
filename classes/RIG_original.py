'''
Original planner for RIG_tree
This program generates an asymptotically optimal informative path planning, rapidly exploring random tree RRT*

ADAPTED FROM - https://www.linkedin.com/pulse/motion-planning-algorithm-rrt-star-python-code-md-mahbubur-rahman/
'''

import os
import numpy as np
from math import atan2, cos, sin
import matplotlib.pyplot as plt

from classes.env.Gaussian2D import *
from classes.env.gp_ipp import GaussianProcessForIPP


class Node:
    def __init__(self, xcoord, ycoord):
        self.x = xcoord
        self.y = ycoord
        self.cost = 0.0
        self.info = 0.0
        self.std = 1.0
        self.parent = None


class RRT:
    def __init__(self, num_nodes=50, XDIM=1.0, YDIM=1.0, radius=0.5, branch_length=0.05, gp_func=None,
                 gaussian_distrib=None):
        # self.path = f'RRT_results/RRT_star_trees/'
        # if not os.path.exists(self.path):
        #     os.makedirs(self.path)

        self.num_nodes = num_nodes
        self.XDIM = XDIM
        self.YDIM = YDIM
        self.radius = radius  # To look for parent & rewiring
        self.branch_length = branch_length  # distance of newly sampled node from tree
        self.node_coords = np.array([])
        self.gp_ipp = gp_func  # GaussianProcessForIPP(self.node_coords)
        self.underlying_distribution = gaussian_distrib
    #     self.ground_truth = self.get_ground_truth()

    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth

    # Utilities
    def distance(self, node1, node2):  # Pass coordinates
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def step_from_to(self, from_node, to_node):  # Pass coordinates
        ## Modified to always step by branch_length
        theta = atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
        newnode = (from_node[0] + self.branch_length * cos(theta), from_node[1] + self.branch_length * sin(theta))
        if newnode[0] < 0 or newnode[0] > self.XDIM or newnode[1] < 0 or newnode[1] > self.YDIM:
            return None
        else:
            return newnode

        if self.distance(from_node, to_node) < self.branch_length:
            return to_node
        else:
            theta = atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
            return from_node[0] + self.branch_length * cos(theta), from_node[1] + self.branch_length * sin(theta)

    def chooseParent(self, nn, newnode):
        # for p in self.nodes:
        #     if self.distance([p.x, p.y], [newnode.x, newnode.y]) < self.radius and self.distance([p.x, p.y], [newnode.x,
        #                                                                                                       newnode.y]) < \
        #             self.distance([nn.x, nn.y], [newnode.x, newnode.y]):
        #         nn = p
        # # print(f"distance is {self.distance([nn.x, nn.y], [newnode.x, newnode.y])}")
        newnode.cost = nn.cost + self.distance([nn.x, nn.y], [newnode.x, newnode.y])
        newnode.parent = nn
        return newnode, nn

    def reWire(self, newnode):
        for i in range(len(self.nodes)):
            p = self.nodes[i]
            if p != newnode.parent and self.distance([p.x, p.y], [newnode.x,
                                                                  newnode.y]) < self.radius and newnode.cost + self.distance(
                [p.x, p.y], [newnode.x, newnode.y]) < p.cost:
                # Show old lines here ->
                p.parent = newnode
                p.cost = newnode.cost + self.distance([p.x, p.y], [newnode.x, newnode.y])
                self.nodes[i] = p
                # Show new lines here ->

    def findNodeIndex(self, p):
        return np.where((self.nodes == p).all(axis=1))[0][0]

    def prune(self, newnode):
        # for p in self.nodes:
        #     if p.std > newnode.std and p.cost < newnode.cost and p.info > newnode.info:
        #         print("remove the branch")
        #         return True

        return False

    def draw_stuff(self):  # , num, start_node):
        x_vals = []
        y_vals = []
        for each_node in self.nodes:
            x_vals.append(each_node.x)
            y_vals.append(each_node.y)

        plt.figure(1)
        plt.scatter(x_vals[1:], y_vals[1:], color='blue')  # All sampled nodes, in blue
        plt.scatter(x_vals[0], y_vals[0], color='orange')  # Start node, in orange
        plt.show()

    def RRT_planner(self, start_node, iterations=300, info=None):
        counts = 0
        self.nodes = []
        start = Node(start_node[0], start_node[1])
        self.nodes.append(start)

        node_C = np.array([[start.x, start.y]])
        start.info, start.std = self.gp_ipp.flexi_updates(node_C) # Predict info

        goal = Node(1.0, 1.0)  # Destination

        while counts < iterations:
            # for i in range(self.num_nodes):
            rand = Node(np.random.rand() * self.XDIM, np.random.rand() * self.YDIM)
            nn = self.nodes[0]
            for p in self.nodes:
                if self.distance([p.x, p.y], [rand.x, rand.y]) < self.distance([nn.x, nn.y], [rand.x, rand.y]):
                    nn = p
            interpolatedNode = self.step_from_to([nn.x, nn.y], [rand.x, rand.y])
            if interpolatedNode is None:
                continue
            newnode = Node(interpolatedNode[0], interpolatedNode[1])
            distance = self.distance([nn.x, nn.y], [newnode.x, newnode.y])
            # print(f"distance is {distance}")
            node_C = np.array([[newnode.x, newnode.y]])
            newnode.info, newnode.std = self.gp_ipp.flexi_updates(node_C)
            # print(f"info is {newnode.info}", f"std is {newnode.std}")
            if not self.prune(newnode):
                [newnode, nn] = self.chooseParent(nn, newnode)
                self.nodes.append(newnode)
                # self.reWire(newnode)

            counts += 1
        #            if counts == iterations:
        #                print('Tree constructed')
        #                self.draw_stuff()
        return self.nodes

    def plot(self, trajectory, prior_position, agent_ID, path_length, all_trajectory):
        x_vals = []
        y_vals = []
        edge_x = []
        edge_y = []
        for each_node in self.nodes[1:]:
            x_vals.append(each_node.x)
            y_vals.append(each_node.y)
            # print(f"parent is {[each_node.parent.x, each_node.parent.y]}")
            edge_x.append([each_node.x, each_node.parent.x])
            edge_y.append([each_node.y, each_node.parent.y])
        plt.figure()
        plt.title(f"current agent is {agent_ID}")
        plt.scatter(x_vals[:], y_vals[:], color='blue')  # All sampled nodes, in blue
        plt.scatter(self.nodes[0].x, self.nodes[0].y, color='orange', marker="*", s=30 ** 2)  # Start node, in orange
        for i in range(len(edge_x)):
            plt.plot(edge_x[i], edge_y[i], color="r")
        path_x = []
        path_y = []

        if all_trajectory:
            for t in all_trajectory:
                all_path_x = []
                all_path_y = []
                t.insert(0, [self.nodes[0].x, self.nodes[0].y])
                # print(f"all trajectory are {all_trajectory}")
                for i in range(1, len(t)):
                    all_path_x.append([t[i - 1][0], t[i][0]])
                    all_path_y.append([t[i - 1][1], t[i][1]])
                for i in range(len(all_path_x)):
                    plt.plot(all_path_x[i], all_path_y[i], color="black", linewidth=2)

        if trajectory:
            trajectory.insert(0, [self.nodes[0].x, self.nodes[0].y])
            for i in range(1, len(trajectory)):
                path_x.append([trajectory[i - 1][0], trajectory[i][0]])
                path_y.append([trajectory[i - 1][1], trajectory[i][1]])
            for i in range(path_length, len(path_x)):
                plt.plot(path_x[i], path_y[i], color="black", linewidth=4)
            for i in range(path_length):
                plt.plot(path_x[i], path_y[i], color="purple", linewidth=4)

        prior_x = []
        prior_y = []
        # print(f"prior position is {prior_position}")
        if prior_position:
            for i in range(1, len(prior_position)):
                prior_x.append([prior_position[i - 1][0], prior_position[i][0]])
                prior_y.append([prior_position[i - 1][1], prior_position[i][1]])

        for i in range(len(prior_x)):
            plt.plot(prior_x[i], prior_y[i], color="y", linewidth=4)
        # print(f"trajectory is {trajectory}", f"path is {path_x}")

        if not os.path.exists("rig_tree_plot"):
            os.mkdir("rig_tree_plot")
        if not os.path.exists("rig_tree_plot/RRT"):
            os.mkdir("rig_tree_plot/RRT")
        plt.savefig('rig_tree_plot/RRT/{}_{}.png'.format(len(prior_position), agent_ID), dpi=150)


if __name__ == '__main__':
    rrt_tree = RRT()
    nodes = rrt_tree.RRT_planner(iterations=100)  # 500 iterations
    print(len(nodes))
