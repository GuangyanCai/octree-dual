from dataclasses import dataclass
import numpy as np
import pymorton as pm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

@dataclass
class Node:
    is_leaf: bool = True

class QuadTree:

    def __init__(self, max_levels):
        self.max_levels = max_levels
        self.nodes = {1: Node()}
        tmp1 = (1 << (max_levels + 1)) - 1
        tmp2 = (1 << (max_levels + 1) * 2) - 1
        self.dil_x = pm.interleave2(tmp1, 0)
        self.dil_y = pm.interleave2(0, tmp1)
        self.dil_x_neg = ~self.dil_x & tmp2
        self.dil_y_neg = ~self.dil_y & tmp2

    def point_to_key(self, point):
        if point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
            return 0
        
        bin_point = np.floor(point * 2 ** self.max_levels)

        key = pm.interleave2(int(bin_point[0]), int(bin_point[1]))
        key |= 1 << (2 * self.max_levels)
        return key
    
    def find_point(self, point):
        key = self.point_to_key(point)
        while not self.key_is_leaf(key) and key != 0:
            key >>= 2
        return key
    
    def key_to_lv(self, key):
        return int(np.floor(np.log2(key) / 2))
    
    def key_exists(self, key):
        return key in self.nodes

    def key_is_leaf(self, key):
        return key in self.nodes and self.nodes[key].is_leaf
    
    def subdiv_by_key(self, key):
        if self.key_is_leaf(key):
            lv = self.key_to_lv(key)
            if lv < self.max_levels:
                new_key = key << 2
                for i in range(4):
                    self.nodes[new_key + i] = Node()

                self.nodes[key].is_leaf = False

    def subdiv_by_point(self, point):
        self.subdiv_by_key(self.find_point(point))

    def get_node(self, key):
        lv = self.key_to_lv(key)
        new_key = key & ((1 << (lv * 2)) - 1)
        side = 2 ** (-lv)
        v0 = np.array(pm.deinterleave2(new_key)).astype(float) * side
        v1 = v0 + np.array([side, 0])
        v2 = v0 + np.array([0, side])
        v3 = v0 + np.array([side, side])

        return v0, v1, v2, v3
    
    def get_node_center(self, key):
        lv = self.key_to_lv(key)
        new_key = key & ((1 << (lv * 2)) - 1)
        side = 2 ** (-lv)
        center = (np.array(pm.deinterleave2(new_key)).astype(float) + 0.5) * side
        return center
    
    def get_quad_tree_segments(self):
        segments = []

        for leaf in self.nodes:
            v0, v1, v2, v3 = self.get_node(leaf)
            segments.append([v0.tolist(), v2.tolist()])
            segments.append([v0.tolist(), v1.tolist()])
            segments.append([v1.tolist(), v3.tolist()])
            segments.append([v2.tolist(), v3.tolist()])
        
        return segments
    
    def leaf_to_vert(self, key):
        lv = self.key_to_lv(key)
        lv_k = 1 << 2 * lv
        vert = []
        for i in range(4):
            v_k = (((key | self.dil_x_neg) + (i & self.dil_x)) & self.dil_x) | (((key | self.dil_y_neg) + (i & self.dil_y)) & self.dil_y)
            if (v_k >= (lv_k << 1)) or not ((v_k - lv_k) & self.dil_x) or not ((v_k - lv_k) & self.dil_y):
                vert.append(0)
            else:
                vert.append(v_k << 2 * (self.max_levels - lv))

        return lv, vert
    
    def vert_to_leaf(self, key, lv):
        dkey = key >> 2 * (self.max_levels - lv)
        adj_nodes = []
        for i in range(4):
            adj_nodes.append((((dkey & self.dil_x) - (i & self.dil_x)) & self.dil_x) | (((dkey & self.dil_y) - (i & self.dil_y)) & self.dil_y))
        return adj_nodes
    
    def gen_dual(self):
        dual_nodes = []

        for key in self.nodes:
            if not self.nodes[key].is_leaf: continue

            lv, vert = self.leaf_to_vert(key)
            for i in range(4):
                if vert[i] == 0: continue

                adj_nodes = self.vert_to_leaf(vert[i], lv)

                skip = False 

                for j in range(4):
                    if j == i: continue

                    if not self.key_exists(adj_nodes[j]): continue
                    
                    if not self.key_is_leaf(adj_nodes[j]) or j < i:
                        skip = True
                        break

                if skip: continue

                for j in range(4):
                    while adj_nodes[j] != 0 and not self.key_exists(adj_nodes[j]):
                        adj_nodes[j] >>= 2

                dual_nodes.append(adj_nodes)

        return dual_nodes

def plot_quad_tree(quad_tree):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.add_collection(LineCollection(quad_tree.get_quad_tree_segments(), colors='blue', linewidths=1.5, linestyle='solid'))

    dual_segments = []
    for node in quad_tree.gen_dual():
        vert = []
        for i in range(4):
            vert.append(quad_tree.get_node_center(node[i]))

        dual_segments.append([vert[0].tolist(), vert[1].tolist()])
        dual_segments.append([vert[1].tolist(), vert[3].tolist()])
        dual_segments.append([vert[3].tolist(), vert[2].tolist()])
        dual_segments.append([vert[2].tolist(), vert[0].tolist()])

    ax.add_collection(LineCollection(dual_segments, colors='red', linewidths=1.5, linestyle='dotted'))

    plt.show()


if __name__ == '__main__':
    quadtree = QuadTree(10)

    for i in range(20):
        quadtree.subdiv_by_point(np.random.rand(2))

    plot_quad_tree(quadtree)
