# implement disparity map using graph cut

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from maxflow import Graph
import numpy as np
import cv2
import time


def mean_image(I, win):
    h, w = I.shape
    IM = np.zeros((h, w), dtype=float)
    area = (2 * win + 1) * (2 * win + 1)
    for j in range(h):
        for i in range(w):
            if j - win < 0 or j + win >= h or i - win < 0 or i + win >= w:
                IM[j, i] = 0
                continue
            s = 0
            for y in range(-win, win + 1):
                for x in range(-win, win + 1):
                    s += I[j + y, i + x]
            IM[j, i] = s / area
    return IM


def correlation(I1, I1M, I2, I2M, win, u1, v1, u2, v2):
    c = 0
    area = (2 * win + 1) * (2 * win + 1)
    for y in range(-win, win + 1):
        for x in range(-win, win + 1):
            c += (I1[v1 + y, u1 + x] - I1M[v1, u1]) * \
                (I2[v2 + y, u2 + x] - I2M[v2, u2])
    return c / area


def correl(I1, I1M, I2, I2M, win, u1, v1, u2, v2):
    c = 0
    area = (2 * win + 1) * (2 * win + 1)
    for y in range(-win, win + 1):
        for x in range(-win, win + 1):
            c += (I1[v1 + y, u1 + x] - I1M[v1, u1]) * \
                (I2[v2 + y, u2 + x] - I2M[v2, u2])
    return c / area


def zncc(I1, I1M, I2, I2M, win, u1, v1, u2, v2):
    var1 = correl(I1, I1M, I1, I1M, win, u1, v1, u1, v1)
    if var1 == 0:
        return 0
    var2 = correl(I2, I2M, I2, I2M, win, u2, v2, u2, v2)
    if var2 == 0:
        return 0
    numerator = correl(I1, I1M, I2, I2M, win, u1, v1, u2, v2)
    denominator = np.sqrt(var1 * var2)
    return numerator / denominator


def rho(x):
    if x <= 0.0:
        return 1.0
    elif x < 1.0:
        return np.sqrt(1 - x)
    else:
        return 0


def coord2node(x, y, d, nx, nd):
    num_node = d + x * nd + y * nd * nx
    return num_node


def build_graph(I1, I2, nx, ny, nd, lambda_, win, zoom, dmin, dmax, wcc):
    INF = 1000000
    dx = [+1, 0, -1, 0]  # 4-neighborhood
    dy = [0, -1, 0, +1]
    K = (nd - 1) * 4 * lambda_
    I1M = mean_image(I1, win)
    I2M = mean_image(I2, win)

    G = Graph[float]()
    G.add_nodes(nx * ny * nd)

    for i in range(nx):
        x = win + zoom * i
        for j in range(ny):
            y = win + zoom * j
            for d in range(nd):
                node_id = coord2node(i, j, d, nx, nd)

                # Adding edges responsible for V
                for k in range(4):
                    ip = i + dx[k]
                    jp = j + dy[k]
                    if 0 < ip < nx and 0 < jp < ny:
                        node_id_p = coord2node(ip, jp, d, nx, nd)
                        if 0 < node_id_p < nx * ny * nd:
                            G.add_edge(
                                node_id, node_id_p, lambda_, lambda_)

                # Adding the edges responsible for E
                if d == 0:
                    try:
                        G.add_tedge(node_id, INF, 0)
                    except:
                        print(node_id)
                elif d == nd - 1:
                    G.add_tedge(node_id, 0, INF)

                if node_id + 1 < nx * ny * nd:
                    if x + win + dmax < I1.shape[1]:
                        D = wcc * rho(zncc(I1, I1M, I2, I2M, win,
                                      x, y, x + d + dmin, y)) + K
                        G.add_edge(node_id, node_id + 1,
                                   D, 0)
    return G


def disparity_map(I1, I2, nx, ny, nd, lambda_, win, zoom, dmin, dmax, wcc):
    graph_start = time.time()
    G = build_graph(I1, I2, nx, ny, nd, lambda_, win,
                    zoom, dmin, dmax, wcc)
    graph_end = time.time()
    print(f"Graph built in{graph_end - graph_start}")
    G.maxflow()
    disparity_map = np.zeros((ny, nx), dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            for d in range(nd):
                node_id = coord2node(i, j, d, nx, nd)
                if G.get_segment(node_id):
                    disparity_map[j, i] = d
                    break

    return disparity_map + dmin




def plot_disparity_map(disparity_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, disparity_map.shape[1], 1)
    y = np.arange(0, disparity_map.shape[0], 1)
    X, Y = np.meshgrid(x, y)
    Z = disparity_map
    ax.plot_surface(X, Y, Z)
    plt.show()


if __name__ == "__main__":

    I1 = cv2.imread(
        '/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/run-L.jpeg', cv2.IMREAD_GRAYSCALE)
    I2 = cv2.imread(
         '/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/run-R.jpeg', cv2.IMREAD_GRAYSCALE)

    
    h1, w1 = I1.shape
    h2, w2 = I2.shape
    
    #crop both images to the same size and square
    if h1 > w1:
        I1 = I1[0:w1, 0:w1]
        I2 = I2[0:w1, 0:w1]
    else:
        I1 = I1[0:h1, 0:h1]
        I2 = I2[0:h1, 0:h1]

    h1, w1 = I1.shape
    h2, w2 = I2.shape
    
    lambdaf = 0.1
    win = (3 - 1) // 2
    zoom = 2
    dmin = 1
    dmax = 40
    wcc = max(1 + int(1 / lambdaf), 20)
    lambda_ = lambdaf * wcc

    nx = (w1 - 2 * win) // zoom
    ny = (h1 - 2 * win) // zoom
    nd = dmax - dmin

    start = time.time()
    disparity_map = disparity_map(
        I1, I2, nx, ny, nd, lambda_, win, zoom, dmin, dmax, wcc)
    end = time.time()

    print("Time: ", end - start)
    plot_disparity_map(disparity_map)
    #plot disparity map as a 2D image
    plt.imshow(disparity_map)
    plt.show()

