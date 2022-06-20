# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-06-14T11:41:47+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-14T22:33:42+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin

import numpy as np


class Node(object):
    def __init__(self, operation, outputs):
        self.operation = operation # Pytorch layer
        self.outputs = outputs # list(Node) ou [None] si output

    def __str__(self):
        out = f"{self.operation} -> {[c.operation for c in self.outputs if c != None]}\n"
        for c in self.outputs:
            if c != None:
                out += c.__str__()

        return out

    def __repr__(self):
        return self.operation

def crossover(root1, root2):
    subset_1 = []
    subset_2 = []

    # Select a connected path from root to output
    current_node = root1
    while current_node.outputs[0] != None:
        idx = np.random.randint(len(current_node.outputs))
        current_node = current_node.outputs[idx]
        subset_1.append(idx)

    current_node = root2
    while current_node.outputs[0] != None:
        idx = np.random.randint(len(current_node.outputs))
        current_node = current_node.outputs[idx]
        subset_2.append(idx)

    #####################
    # PARENT 1
    #####################

    # Select starting and ending nodes for sub-tree 1
    # index_1_1 = where we cut the link, between 2 nodes in tree 1
    # previous_node_1_1 = the first node before the cut
    # The second node after the cut will be the starting point of sub-tree 1
    # index_1_2 = where we cut the link, between 2 nodes in tree 1
    # The first node will be the ending point of sub-tree 1
    index_1_1 = np.random.randint(len(subset_1)-1)
    if index_1_1+1 < len(subset_1)-1:
        index_1_2 = np.random.randint(index_1_1+1,len(subset_1)-1)
    else:
        index_1_2=index_1_1

    # Go from the input to the output of sub-tree 1, to retrieve starting and ending nodes
    current_node = root1
    previous_node_1_1 = None
    for idx in range(index_1_1+1):
        previous_node_1_1 = current_node
        current_node = current_node.outputs[subset_1[idx]]

    if index_1_1 == index_1_2:
        end_1=None
    else:
        current_node = root1
        previous_node_1_2 = None
        for idx in range(index_1_2+1):
            previous_node_1_2 = current_node
            current_node = current_node.outputs[subset_1[idx]]
        end_1 = current_node

    # remove starting point from the previous node
    start_1 = previous_node_1_1.outputs.pop(subset_1[index_1_1])

    #####################
    # PARENT 2
    #####################


    # Select starting and ending nodes for sub-tree 2
    # previous_node_2_1 = the first node before the cut
    # The second node after the cut will be the starting point of sub-tree 1
    # index_2_2 = where we cut the link, between 2 nodes in tree 2
    # The first node will be the ending point of sub-tree 2
    index_2_1 = np.random.randint(len(subset_2)-1)
    if index_2_1+1 < len(subset_2)-1:
        index_2_2 = np.random.randint(index_2_1+1,len(subset_2)-1)
    else:
        index_2_2=index_2_1

    # Go from the input to the output of sub-tree 1, to retrieve starting and ending nodes
    current_node = root2
    previous_node_2_1 = None
    for idx in range(index_2_1+1):
        previous_node_2_1 = current_node
        current_node = current_node.outputs[subset_2[idx]]

    if index_2_1 == index_2_2:
        end_2=None
    else:
        current_node = root2
        previous_node_2_2 = None
        for idx in range(index_2_2+1):
            previous_node_2_2 = current_node
            current_node = current_node.outputs[subset_2[idx]]
        end_2 = current_node

    # remove starting point from the previous node
    start_2 = previous_node_2_1.outputs.pop(subset_2[index_2_1])

    #####################
    # Crossover
    #####################

    # Remove all nodes attached to sub-tree 1
    # store them into outputs of sub-tree 2
    if end_1:
        current_node = start_1
        outputs_2 = []
        for idx in subset_1[index_1_1+1:index_1_2+1]:
            outputs_2 += current_node.outputs[:idx]+current_node.outputs[idx+1:]
            current_node.outputs = [current_node.outputs[idx]]
            current_node = current_node.outputs[0]
    else:
        outputs_2 = start_1.outputs

    # Remove all nodes attached to sub-tree 2
    # store them into outputs of sub-tree 1
    if end_2:
        current_node = start_2
        outputs_1 = []
        for idx in subset_2[index_2_1+1:index_2_2+1]:
            outputs_1 += current_node.outputs[:idx]+current_node.outputs[idx+1:]
            current_node.outputs = [current_node.outputs[idx]]
            current_node = current_node.outputs[0]
    else:
        outputs_1 = start_2.outputs

    # Replace outputs of sub-tree 1 by output of sub-tree 2 and converse
    previous_node_1_1.outputs.append(start_2)
    previous_node_2_1.outputs.append(start_1)

    if end_1:
        end_1.outputs = outputs_1
    else:
        start_1.outputs = outputs_1

    if end_2:
        end_2.outputs = outputs_2
    else:
        start_2.outputs = outputs_2
