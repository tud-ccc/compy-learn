import collections

import networkx as nx
import pygraphviz as pgv


class RepresentationBuilder(object):
    def __init__(self):
        self._tokens = collections.OrderedDict()

    def num_tokens(self):
        return len(self._tokens)

    def get_tokens(self):
        return list(self._tokens.keys())

    def print_tokens(self):
        print("-" * 50)
        print("{:<8} {:<25} {:<10}".format("NodeID", "Label", "Number"))
        t_view = [(v, k) for k, v in self._tokens.items()]
        t_view = sorted(t_view, key=lambda x: x[0], reverse=True)
        for v, k in t_view:
            idx = list(self._tokens.keys()).index(k)
            print("{:<8} {:<25} {:<10}".format(str(idx), str(k), str(v)))
        print("-" * 50)


class Sequence(object):
    def __init__(self, S, token_types):
        self.S = S
        self.__token_types = token_types

    def get_token_list(self):
        node_ints = [self.__token_types.index(token_str) for token_str in self.S]

        return node_ints

    def size(self):
        return len(self.S)

    def draw(self, width=8, limit=30, path=None):
        # Create dot graph.
        graphviz_graph = pgv.AGraph(
            directed=True,
            splines=False,
            rankdir="LR",
            nodesep=0.001,
            ranksep=0.4,
            outputorder="edgesfirst",
            fillcolor="white",
        )

        remaining_tokens = None
        for i, token in enumerate(self.S):
            if i == limit:
                remaining_tokens = 5

            if remaining_tokens is not None:
                if remaining_tokens > 0:
                    token = "..."
                    remaining_tokens -= 1
                else:
                    break

            if i % width == 0:
                subgraph = graphviz_graph.subgraph(
                    name="cluster_%i" % i, label="", color="white"
                )

                graphviz_graph.add_node(i, label=token, shape="box")
                if i > 0:
                    graphviz_graph.add_edge(
                        i - width, i, color="white", constraint=False
                    )
            else:
                subgraph.add_node(i, label=token, shape="box")
            if i > 0:
                if i % width == 0:
                    graphviz_graph.add_edge(i - 1, i, constraint=False, color="gray")
                else:
                    graphviz_graph.add_edge(i - 1, i)

        graphviz_graph.layout("dot")

        return graphviz_graph.draw(path)


class Graph(object):
    def __init__(self, graph, node_types, edge_types):
        self.G = graph
        self.__node_types = node_types
        self.__edge_types = edge_types

    def _get_node_attr_dict(self):
        return collections.OrderedDict(self.G.nodes(data="attr", default="N/A"))

    def get_node_str_list(self):
        node_strs = list(self._get_node_attr_dict().values())

        return node_strs

    def get_node_list(self):
        node_strs = list(self._get_node_attr_dict().values())
        node_ints = [self.__node_types.index(node_str) for node_str in node_strs]

        return node_ints

    def get_edge_list(self):
        nodes_keys = list(self._get_node_attr_dict().keys())

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edges.append(
                (
                    nodes_keys.index(node1),
                    self.__edge_types.index(data["attr"]),
                    nodes_keys.index(node2),
                )
            )

        return edges

    def get_leaf_node_list(self):
        """Return an ordered list of node indices for leaves of the graph.

        Only useful for graphs that are built based on a sequence (like ASTs on tokens)
        """
        nodes_keys = list(self._get_node_attr_dict().keys())

        data = { n: order for n, order in self.G.nodes(data='seq_order') if order is not None }
        return [nodes_keys.index(n) for n, _ in sorted(data.items(), key=lambda x: x[1])]

    def map_to_leaves(self, relations=None):
        """Map inner nodes of the graph to leaf nodes.

        Leaves are all nodes which have a `seq_order` attribute.
        Each node in the graph which is not a leaf is mapped to the descendant leaf with lowest seq_order.
        All edges are moved to the mapped leaf nodes.
        Nodes which have no leaf descendants are not transformed.

        The `relations` parameter specifies which edges indicate a parent-child relationship.
        The value of this parameter must be a dict with two keys, `child` and `parent`.
        For both keys, the value must be a collection of edge types of that kind.

        Returns the new graph.
        """
        if relations is None:
            relations = {
                'child': {'ast', 'token'},
                'parent': {'in', 'data'},
            }
        relations.setdefault('parent', set())
        relations.setdefault('child', set())

        result = nx.MultiDiGraph()

        # Map nodes to leaf node
        leaf_for_node = {}

        # Walk leaves in sequential order, so sequentially first leaves get priority
        leaves = { n: data for n, data in self.G.nodes(data=True) if 'seq_order' in data }
        for leaf, data in sorted(leaves.items(), key=lambda x: x[1]['seq_order']):
            result.add_node(leaf, **data)
            # Walk the tree upwards, and assign any unassigned nodes to this leaf
            todo = { leaf }
            while todo:
                n = todo.pop()
                if n in leaf_for_node and leaf_for_node[n][1] <= data['seq_order']:
                    # this node has already been assigned to an earlier leaf, so no need to traverse further
                    continue
                leaf_for_node[n] = (leaf, data['seq_order'])
                todo.update(set(target for _, target, attr in self.G.out_edges(n, data='attr') if attr in relations['parent']))
                todo.update(set(source for source, _, attr in self.G.in_edges(n, data='attr') if attr in relations['child']))
            # ensure leaves are never mapped to other leaves
            # this makes sure that the function is idempotent
            leaf_for_node[leaf] = (leaf, data['seq_order'])

        # Translate edges
        for source, target, data in self.G.edges(data=True):
            source = leaf_for_node.get(source, (source, 0))[0]
            target = leaf_for_node.get(target, (target, 0))[0]

            if source in leaf_for_node:
                source = leaf_for_node[source][0]
            else:
                result.add_node(source, **self.G.nodes(data=True)[source])

            if target in leaf_for_node:
                target = leaf_for_node[target][0]
            else:
                result.add_node(target, **self.G.nodes(data=True)[target])

            result.add_edge(source, target, **data)

        return Graph(result, list(self.__node_types), list(self.__edge_types))

    def size(self):
        return len(self.G)

    def draw(self, path=None, with_legend=False, align_tokens=True):
        # Copy graph object because attr modifications for a cleaner view are needed.
        G = self.G

        # Add node labels.
        for (n, data) in G.nodes(data=True):
            if "attr" in data:
                if type(data["attr"]) is tuple:
                    label = "\n".join(data["attr"])
                else:
                    label = data["attr"]

                G.nodes[n]["label"] = label

        # Add edge colors.
        edge_colors_by_types = {
            "ast": "black",
            "cfg": "green",
            "data": "blue",
            "mem": "pink",
            "call": "yellow",
        }
        edge_colors_available = ["orange", "pink", "cyan", "crimson", "darkgreen", "darkblue", "darkcyan"]
        for etype in self.__edge_types:
            if etype in edge_colors_by_types: continue
            edge_colors_by_types[etype] = edge_colors_available.pop(0)

        for u, v, key, data in G.edges(keys=True, data=True):
            edge_type = data["attr"]
            if edge_type not in edge_colors_by_types:
                edge_colors_by_types[edge_type] = edge_colors_available.pop(0)

            G[u][v][key]["color"] = edge_colors_by_types[edge_type]

            # G[u][v][key]['weight'] = 10 if edge_type == 'cfg' else 0

        # Create dot graph.
        graphviz_graph = nx.drawing.nx_agraph.to_agraph(G)

        # Add Legend.
        if with_legend:
            edge_types_used = set()
            for (u, v, key, data) in G.edges(keys=True, data=True):
                edge_type = data["attr"]
                edge_types_used.add(edge_type)

            subgraph = graphviz_graph.subgraph(name="cluster", label="Edges")
            for edge_type, color in edge_colors_by_types.items():
                if edge_type in edge_types_used:
                    subgraph.add_node(edge_type, color="invis", fontcolor=color)

        # Put all tokens on single level ("rank") and enforce order
        if align_tokens:
            tokens = graphviz_graph.subgraph(rank="sink", rankdir="LR")
            leaves = { n: data for n, data in self.G.nodes(data=True) if 'seq_order' in data }
            leaf_nodes = list(sorted(leaves.items(), key=lambda x: x[1]['seq_order']))
            for a, b in zip(leaf_nodes, leaf_nodes[1:]) :
                tokens.add_edge(a[0], b[0], color="invis")

        graphviz_graph.layout("dot")
        return graphviz_graph.draw(path)
