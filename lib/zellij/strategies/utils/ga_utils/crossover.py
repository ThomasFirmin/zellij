def DAGraphCrossover(g1, g2):
    graph1 = g1.copy()
    graph2 = g2.copy()
    subset_1 = []
    subset_2 = []

    # Select a connected path from root to output
    current_node = graph1.root
    while len(current_node.outputs) != 0:
        idx = np.random.randint(len(current_node.outputs))
        current_node = current_node.outputs[idx]
        subset_1.append(current_node)

    current_node = graph2.root
    while len(current_node.outputs) != 0:
        idx = np.random.randint(len(current_node.outputs))
        current_node = current_node.outputs[idx]
        subset_2.append(current_node)

    index_1_1 = np.random.randint(len(subset_1))
    if index_1_1 + 1 < len(subset_1):
        index_1_2 = np.random.randint(index_1_1 + 1, len(subset_1))
    else:
        index_1_2 = index_1_1

    index_2_1 = np.random.randint(len(subset_2))
    if index_2_1 + 1 < len(subset_2):
        index_2_2 = np.random.randint(index_2_1 + 1, len(subset_2))
    else:
        index_2_2 = index_2_1
    subset_1 = subset_1[index_1_1:index_1_2 + 1]
    subset_2 = subset_2[index_2_1:index_2_2 + 1]

    child1 = [n for n in graph1.nodes if n not in subset_1]
    child2 = [n for n in graph2.nodes if n not in subset_2]

    subset_1[-1].outputs, subset_2[-1].outputs = subset_2[-1].outputs, subset_1[-1].outputs
    orphan_nodes_1 = []
    orphan_nodes_2 = []
    if len(subset_1) > 1:
        for i in range(len(subset_1) - 1, 0, -1):
            orphan_nodes_1 += remove_node_from_list(subset_1[i - 1].outputs, subset_1[i])
            subset_1[i - 1].outputs = [subset_1[i]]
    orphan_nodes_1 = list(dict.fromkeys(orphan_nodes_1))
    if len(subset_2) > 1:
        for i in range(len(subset_2) - 1, 0, -1):
            orphan_nodes_2 += remove_node_from_list(subset_2[i - 1].outputs, subset_2[i])
            subset_2[i - 1].outputs = [subset_2[i]]
    orphan_nodes_2 = list(dict.fromkeys(orphan_nodes_2))

    for i in range(len(child1)):
        if child1[i].is_in_outputs(subset_1[0]):
            child1[i].outputs = remove_node_from_list(child1[i].outputs, subset_1[0]) + [subset_2[0]] + orphan_nodes_1
        if child1[i].is_in_outputs(subset_1[-1]):
            child1[i].outputs = remove_node_from_list(child1[i].outputs, subset_1[-1]) + [subset_2[-1]]
    child1 += subset_2
    for i in range(len(child2)):
        if child2[i].is_in_outputs(subset_2[0]):
            child2[i].outputs = remove_node_from_list(child2[i].outputs, subset_2[0]) + [subset_1[0]] + orphan_nodes_2
        if child2[i].is_in_outputs(subset_2[-1]):
            child2[i].outputs = remove_node_from_list(child2[i].outputs, subset_2[-1]) + [subset_1[-1]]
    child2 += subset_1

    child1 = DAGraph(child1)
    child2 = DAGraph(child2)
    return child1, child2