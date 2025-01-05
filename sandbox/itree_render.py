import subprocess
from intervaltree import IntervalTree
from anytree import RenderTree, Node as VisualNode
from intervaltree.node import Node
from anytree.exporter import DotExporter
from typing import cast


def get_traverse_tree(node: Node) -> VisualNode:
    s = []
    for interval in sorted(node.s_center):
        s.append(str(interval.data))
    s = "{" + " AND ".join(s) + "}"

    if not node.left_node and not node.right_node:
        return VisualNode(s)

    children = []
    if node.left_node:
        children.append(get_traverse_tree(node.left_node))
    if node.right_node:
        children.append(get_traverse_tree(node.right_node))

    return VisualNode(s, children=children)


def render_tree(itree: IntervalTree):
    root = get_traverse_tree(cast(Node, itree.top_node))

    # Export to a .dot file for Graphviz
    DotExporter(root).to_dotfile("tree.dot")

    command = "dot -Tpng  tree.dot > tree.png"
    subprocess.run(command, shell=True)

    return 'tree.png'


def render_tree_str(itree: IntervalTree) -> RenderTree:
    return RenderTree(get_traverse_tree(cast(Node, itree.top_node)))
