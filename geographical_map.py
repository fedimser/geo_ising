from collections import Counter

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import shapely.geometry as geom
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from matplotlib.collections import PatchCollection, LineCollection

from ising_model import IsingModel


class Edge:
    def __init__(self, v1: int, v2: int, type: str, value: float):
        if v1 < v2: v1, v2 = v2, v1
        self.v1 = v1
        self.v2 = v2
        self.type = type
        self.value = value

    def other_vertex(self, v: int) -> int:
        if self.v1 == v:
            return self.v2
        return self.v1


class GeographicalMap:
    def __init__(self, region_ids, polygons):
        # Initialize regions.
        assert len(region_ids) == len(set(region_ids))
        self.region_ids = np.array(region_ids)
        self.N = self.region_ids.shape[0]
        self._region_rev_index = {}
        for i in range(self.N):
            self._region_rev_index[self.region_ids[i]] = i

        # Initialize polygons.
        self.polygons = [np.array(p) for p in polygons]
        self.shapely_polygons = [geom.Polygon(p) for p in polygons]

        # Calculate centroids.
        self.centroids = np.zeros((self.N, 2))
        for i in range(self.N):
            self.centroids[i, :] = self.shapely_polygons[i].centroid
        self._centroids_kd_tree = cKDTree(self.centroids)

        # Calculate bounding box.
        all_coords = np.concatenate(self.polygons, axis=0)
        self.xmin = np.min(all_coords[:, 0])
        self.xmax = np.max(all_coords[:, 0])
        self.ymin = np.min(all_coords[:, 1])
        self.ymax = np.max(all_coords[:, 1])

        self.population = np.zeros(self.N, dtype=np.int64)

        self.edges = dict()
        self.edge_lists = [[] for i in range(self.N)]

    def set_population(self, region_id, pop):
        self.population[self._region_rev_index[region_id]] = pop

    def get_point_region(self, x, y, max_dist=3.0):
        """Returns region containing given point."""
        idx = self._centroids_kd_tree.query_ball_point([x, y], max_dist)
        for region_id in idx:
            if self.shapely_polygons[region_id].contains(geom.Point(x, y)):
                return self.region_ids[region_id]
        return None

    def _add_edge_by_index(self, v1: int, v2: int, type: str, value: float):
        assert v1 != v2
        assert (v1, v2) not in self.edges
        edge = Edge(v1, v2, type, value)
        self.edges[(v1, v2)] = edge
        self.edges[(v2, v1)] = edge
        self.edge_lists[v1].append(edge)
        self.edge_lists[v2].append(edge)

    def add_edge(self, region1: str, region2: str, type: str, value: float):
        assert self.has_region(region1) and self.has_region(region2)
        v1 = self._region_rev_index[region1]
        v2 = self._region_rev_index[region2]
        self._add_edge_by_index(v1, v2, type, value)

    def get_edge_by_index(self, v1: int, v2: int):
        if (v1, v2) not in self.edges:
            return None
        return self.edges[(v1, v2)]

    def get_edge(self, region1: str, region2: str):
        assert self.has_region(region1) and self.has_region(region2)
        v1 = self._region_rev_index[region1]
        v2 = self._region_rev_index[region2]
        return self.get_edge_by_index(v1, v2)

    def calculate_land_edges(self):
        assert len(self.edges) == 0

        # For each corner find all regions to which it belongs.
        point_to_region = dict()
        for region_id in range(self.N):
            for point in self.polygons[region_id]:
                x = int(point[0] * 100)
                y = int(point[1] * 100)
                point_id = (x, y)
                if point_id in point_to_region:
                    point_to_region[point_id].append(region_id)
                else:
                    point_to_region[point_id] = [region_id]

        # Calculate all edges connecting neighboring regions.
        # Regions are neighbors if they have at least 2 common corners.
        edges = Counter()
        for _, region_list in point_to_region.items():
            adj_regions = list(set(region_list))
            for v1 in adj_regions:
                for v2 in adj_regions:
                    if v1 < v2: edges[(v1, v2)] += 1
        for v1, v2 in edges:
            if edges[(v1, v2)] >= 2:
                self._add_edge_by_index(v1, v2, 'land', 1.0)

    def has_region(self, region_id: str):
        return region_id in self._region_rev_index

    def get_neighbors(self, v: int, edge_type=None):
        """Returns indices of regions incident to given region."""
        ans = []
        for edge in self.edge_lists[v]:
            if edge_type is not None and edge.type != edge_type: continue
            ans.append(edge.other_vertex(v))
            ans.sort()
        return ans


def draw_map(map: GeographicalMap, mode='random', data=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(map.xmin, map.xmax)
    ax.set_ylim(map.ymin, map.ymax)

    patches = PatchCollection([
        matplotlib.patches.Polygon(p, fill=False) for p in map.polygons
    ])

    to_line = lambda e: map.centroids[(e.v1, e.v2), :]
    land_edges = [to_line(e) for e in map.edges.values() if e.type == 'land']
    air_edges = [to_line(e) for e in map.edges.values() if e.type == 'air']
    land_edges = LineCollection(land_edges, linewidths=0.3, colors='red')
    air_edges = LineCollection(air_edges, linewidths=0.1, colors='green')

    ax.add_collection(patches)

    if mode == 'random':
        patches.set_array(np.random.randint(0, 20, size=map.N))
        patches.set_cmap(matplotlib.cm.jet)
    elif mode == 'population':
        patches.set_array(map.population)
        patches.set_cmap(matplotlib.cm.jet)
        patches.set_norm(matplotlib.colors.LogNorm())
        plt.colorbar(patches, ax=ax)
    elif mode == 'graph':
        patches.set_color('black')
        patches.set_facecolor('white')
        patches.set_linewidth(0.1)
        ax.scatter(map.centroids[:, 0], map.centroids[:, 1], s=5)

        # Plot edges.
        ax.add_collection(land_edges)
        ax.add_collection(air_edges)
    elif mode == 'data':
        patches.set_array(data)
        patches.set_cmap(matplotlib.cm.jet)
        plt.colorbar(patches, ax=ax)

    plt.show()


def to_planar_ising_model(gmap: GeographicalMap):
    from third_party.planar_ising.planar_ising import PlanarIsingModel, \
        PlanarGraphConstructor

    adj = [gmap.get_neighbors(i, edge_type='land') for i in range(map.N)]

    graph = PlanarGraphConstructor.construct_from_ordered_adjacencies(adj)
    interactions = np.zeros(graph.edges_count)
    for i in range(graph.edges_count):
        v1 = graph.edges.vertex1[i]
        v2 = graph.edges.vertex2[i]
        edge = gmap.get_edge_by_index(v1, v2)
        if edge is not None:
            interactions[i] = edge.value

    return PlanarIsingModel(graph, interactions)
