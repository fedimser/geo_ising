import numpy as np

from geographical_map import GeographicalMap


def test_land_edges_grid_2x2():
    ids = ["00", "01", "10", "11"]
    coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
    polygons = [np.array([[x, y], [x, y + 1], [x + 1, y + 1], [x + 1, y]]) for
                x, y in coords]

    gmap = GeographicalMap(ids, polygons)
    gmap.calculate_land_edges()
    assert gmap.get_neighbors(0) == [1, 2]
    assert gmap.get_neighbors(1) == [0, 3]
    assert gmap.get_neighbors(2) == [0, 3]
    assert gmap.get_neighbors(3) == [1, 2]
