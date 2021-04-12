import networkx as nx

import compy.representations.common as common


def sample_graph():
    G = nx.MultiDiGraph()
    G.add_node("root1", attr="root")
    for n in ["n1", "n2", "n3", "n4", "n5"]:
        G.add_node(n, attr=n)
    for n in ["n1", "n2", "n3"]:
        G.add_edge("root1", n, attr="child")
    G.add_edge("n4", "n3", attr="parent")
    G.add_edge("n4", "n5", attr="child")
    for l in range(7):
        G.add_node("l" + str(l+1), attr="leaf" + str(l+1), seq_order=l)
    G.add_edge("n1", "l1", attr="token")
    G.add_edge("n1", "l2", attr="token")
    G.add_edge("n2", "l3", attr="token")
    G.add_edge("root1", "l4", attr="token")
    G.add_edge("n4", "l5", attr="token")
    G.add_edge("n4", "l6", attr="token")
    G.add_edge("n4", "l7", attr="token")
    G.add_edge("root1", "n3", attr="rel2")
    G.add_edge("l1", "l2", attr="token_rel")
    G.add_edge("n2", "l2", attr="node_token_rel")
    return common.Graph(G,
                        list(sorted(set(attr for _, attr in G.nodes(data="attr")))),
                        list(sorted(set(attr for _, _, attr in G.edges(data="attr")))))


def test_map_to_leaves():
    graph = sample_graph()
    relations = {'child': {'token', 'child'}, 'parent': {'parent'}}
    leaves_only = graph.map_to_leaves(relations)

    # n5 is kept because it has no child nodes
    assert sorted(leaves_only.get_node_str_list()) == ['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7', 'n5']

    # map to leaves should be idempotent
    assert sorted(graph.map_to_leaves(relations).G.edges(data='attr')) == sorted(
        leaves_only.map_to_leaves(relations).G.edges(data='attr'))

    edges = sorted(leaves_only.G.edges(data='attr'))
    expected_edges = [
        ('l1', 'l1', 'token'),
        ('l1', 'l2', 'token'),
        ('l3', 'l3', 'token'),
        ('l1', 'l4', 'token'),
        ('l5', 'l5', 'token'),
        ('l5', 'l6', 'token'),
        ('l5', 'l7', 'token'),
        ('l1', 'l1', 'child'),
        ('l1', 'l3', 'child'),
        ('l1', 'l5', 'child'),
        ('l5', 'n5', 'child'),
        ('l1', 'l2', 'token_rel'),
        ('l3', 'l2', 'node_token_rel'),
        ('l1', 'l5', 'rel2'),
        ('l5', 'l5', 'parent'),
    ]
    assert edges == sorted(expected_edges)

    without_parent = graph.map_to_leaves({'child': ['token', 'child']})
    assert sorted(without_parent.get_node_str_list()) == ['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7', 'n3', 'n5']

def test_map_to_leaves_cycle():
    cycle = nx.MultiDiGraph()
    for node in ["0", "1", "2", "4"]:
        cycle.add_node(node, attr=node)
    cycle.add_edge("0", "1", attr='flow')
    cycle.add_edge("1", "2", attr='flow')
    cycle.add_edge("2", "0", attr='flow')
    cycle.add_edge("4", "0", attr='flow')
    cycle.add_node('leaf', attr='leaf', seq_order=0)
    cycle.add_edge("0", 'leaf', attr='flow')

    graph = common.Graph(cycle, [], ['leaf', "0", "1", "2", "4"])
    mapped = graph.map_to_leaves({'child': 'flow'})
    assert sorted(mapped.G.edges(data=False)) == [("leaf", "leaf")] * 5
