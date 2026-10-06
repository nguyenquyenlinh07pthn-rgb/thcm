"""
MÔ-ĐUN 3: PHÂN TÍCH MẠNG LƯỚI TƯƠNG TÁC & MÔ HÌNH MARKOV (Bài 6, Bài 7)
Hệ thống Thương mại Điện tử POLY-MART

Nội dung:
- Chức năng 3.1: Biểu diễn Đồ thị, BFS tìm đường đi ngắn nhất & DFS đếm số cụm liên thông.
- Chức năng 3.2: Chuỗi Markov dự báo hành vi mua sắm (k ngày & Trạng thái dừng pi*).
- Chức năng 3.3: Kiểm tra Đồ thị hai phía (Bipartite Graph) bằng thuật toán tô 2 màu BFS.
- Mở rộng Mạng giao vận (DSA):
    + Dijkstra tìm tuyến vận chuyển rẻ nhất (kết hợp heapq).
    + Kruskal + Union-Find (DSU) tối ưu hóa mạng lưới kết nối tối thiểu (MST).
"""

import sys
import heapq
from collections import deque, defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ==============================================================================
# CHỨC NĂNG 3.1: BIỂU DIỄN ĐỒ THỊ & DUYỆT BFS / DFS
# ==============================================================================

class SocialNetworkGraph:
    """
    Quản lý mạng lưới người dùng và kết bạn trong POLY-MART.
    """
    def __init__(self, users):
        self.users = users
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.adj_list = defaultdict(list)
        self.num_users = len(users)

    def add_edge(self, u, v):
        """Thêm cạnh vô hướng giữa 2 người dùng."""
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def to_adjacency_matrix(self):
        """Chuyển đổi danh sách kề thành ma trận kề N x N."""
        n = self.num_users
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for u, neighbors in self.adj_list.items():
            i = self.user_to_idx[u]
            for v in neighbors:
                j = self.user_to_idx[v]
                matrix[i][j] = 1
        return matrix

    def bfs_shortest_path(self, start_user, target_user):
        """
        Dùng BFS tìm đường đi ngắn nhất giữa 2 người dùng trong mạng giới thiệu.
        Trả về danh sách các node trên lộ trình và khoảng cách.
        """
        if start_user not in self.user_to_idx or target_user not in self.user_to_idx:
            return None, -1
        if start_user == target_user:
            return [start_user], 0

        visited = {start_user}
        queue = deque([(start_user, [start_user])])

        while queue:
            current, path = queue.popleft()
            for neighbor in self.adj_list[current]:
                if neighbor == target_user:
                    return path + [neighbor], len(path)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None, -1  # Không có đường đi

    def dfs_count_connected_components(self):
        """
        Dùng DFS đếm số cụm cộng đồng liên thông trong mạng lưới.
        """
        visited = set()
        components = []

        def dfs(node, current_comp):
            visited.add(node)
            current_comp.append(node)
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_comp)

        for u in self.users:
            if u not in visited:
                comp = []
                dfs(u, comp)
                components.append(comp)

        return len(components), components


# ==============================================================================
# CHỨC NĂNG 3.2: CHUỖI MARKOV DỰ BÁO HÀNH VI MUA SẮM
# ==============================================================================

class MarkovChainBuyerBehavior:
    """
    Chuỗi Markov mô hình hóa hành vi khách hàng qua 3 trạng thái:
    0: Xem hàng (Browse)
    1: Mua hàng (Purchase)
    2: Rời đi (Churn/Leave)
    """
    def __init__(self, states, transition_matrix):
        self.states = states
        self.P = transition_matrix  # Ma trận xác suất chuyển trạng thái 3x3
        self.n = len(states)

    def mat_mul(self, A, B):
        """Nhân hai ma trận vuông."""
        res = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                res[i][j] = sum(A[i][k] * B[k][j] for k in range(self.n))
        return res

    def matrix_power(self, M, k):
        """Tính lũy thừa ma trận M^k."""
        res = [[1.0 if i == j else 0.0 for j in range(self.n)] for i in range(self.n)]
        base = M
        power = k
        while power > 0:
            if power % 2 == 1:
                res = self.mat_mul(res, base)
            base = self.mat_mul(base, base)
            power //= 2
        return res

    def predict_state_distribution(self, initial_dist, k_days):
        """
        Dự báo phân phối xác suất trạng thái sau k ngày: v_k = v_0 * P^k.
        """
        P_k = self.matrix_power(self.P, k_days)
        # v_k = initial_dist * P_k
        v_k = [0.0] * self.n
        for j in range(self.n):
            v_k[j] = sum(initial_dist[i] * P_k[i][j] for i in range(self.n))
        return [round(x, 4) for x in v_k]

    def compute_steady_state(self, max_iters=200, tol=1e-7):
        """
        Tìm phân phối dừng pi* (Steady State) thỏa mãn: pi* * P = pi* và sum(pi*) = 1.
        Sử dụng phương pháp lặp phân phối (Power Iteration on Markov Chain).
        """
        # Khởi tạo phân phối đều
        pi = [1.0 / self.n] * self.n
        for _ in range(max_iters):
            next_pi = [0.0] * self.n
            for j in range(self.n):
                next_pi[j] = sum(pi[i] * self.P[i][j] for i in range(self.n))
            # Kiểm tra hội tụ L1
            diff = sum(abs(next_pi[j] - pi[j]) for j in range(self.n))
            pi = next_pi
            if diff < tol:
                break
        return [round(x, 4) for x in pi]


# ==============================================================================
# CHỨC NĂNG 3.3: KIỂM TRA ĐỒ THỊ HAI PHÍA (BIPARTITE GRAPH) BẰNG BFS
# ==============================================================================

def is_bipartite_graph(graph_adj):
    """
    Kiểm tra mạng lưới ghép nối (Kho hàng - Cửa hàng) có phải đồ thị hai phía hay không
    bằng thuật toán tô 2 màu BFS (2-Coloring).
    
    Tham số:
        graph_adj (dict): Danh sách kề biểu diễn đồ thị.
        
    Trả về:
        tuple: (is_bipartite, color_map)
    """
    color = {}  # 0 hoặc 1

    for start_node in graph_adj:
        if start_node not in color:
            color[start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                for v in graph_adj[u]:
                    if v not in color:
                        # Gán màu đối lập cho đỉnh kề
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        # Cùng màu => Không thể là đồ thị hai phía!
                        return False, {}

    return True, color


# ==============================================================================
# PHẦN MỞ RỘNG DSA: MẠNG GIAO VẬN ROUTING (DIJKSTRA & KRUSKAL DSU)
# ==============================================================================

class DisjointSetUnion:
    """Cấu trúc dữ liệu Disjoint Set Union (Union-Find) tối ưu Path Compression và Union by Rank."""
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False


def kruskal_minimum_spanning_tree(nodes, edges):
    """
    Thuật toán Kruskal tìm Cây khung nhỏ nhất (MST) kết nối toàn bộ hệ thống kho hàng
    với chi phí đường dây/tuyến đường rẻ nhất.
    edges = [(chi_phí, u, v), ...]
    """
    edges_sorted = sorted(edges, key=lambda x: x[0])
    dsu = DisjointSetUnion(nodes)
    mst = []
    total_cost = 0

    for weight, u, v in edges_sorted:
        if dsu.union(u, v):
            mst.append((u, v, weight))
            total_cost += weight
            if len(mst) == len(nodes) - 1:
                break

    return total_cost, mst


def dijkstra_cheapest_route(graph_weighted, start_node, target_node):
    """
    Thuật toán Dijkstra tìm tuyến vận chuyển rẻ nhất từ điểm giao hàng start đến target.
    graph_weighted = {u: [(v, weight), ...]}
    """
    distances = {node: float('inf') for node in graph_weighted}
    distances[start_node] = 0
    pq = [(0, start_node, [start_node])]

    while pq:
        curr_dist, curr_node, path = heapq.heappop(pq)

        if curr_node == target_node:
            return curr_dist, path

        if curr_dist > distances[curr_node]:
            continue

        for neighbor, weight in graph_weighted[curr_node]:
            dist = curr_dist + weight
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                heapq.heappush(pq, (dist, neighbor, path + [neighbor]))

    return float('inf'), []


# ==============================================================================
# HÀM DEMO MODULE 3
# ==============================================================================

def demo_module_3():
    """Hàm chạy demo kiểm thử toàn bộ Chức năng Module 3."""
    print("=" * 70)
    print("DEMO MÔ-ĐUN 3: PHÂN TÍCH MẠNG LƯỚI TƯƠNG TÁC & MÔ HÌNH MARKOV")
    print("=" * 70)

    # 1. Đồ thị mạng lưới xã hội người dùng
    users = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7']
    social_net = SocialNetworkGraph(users)
    connections = [
        ('U1', 'U2'), ('U2', 'U3'), ('U1', 'U3'),
        ('U3', 'U4'),
        ('U5', 'U6')  # Cụm riêng lẻ
        # U7 độc lập
    ]
    for u, v in connections:
        social_net.add_edge(u, v)

    print("\n--- 3.1. DUYỆT ĐỒ THỊ MẠNG XÃ HỘI (BFS / DFS) ---")
    path, dist = social_net.bfs_shortest_path('U1', 'U4')
    print(f"Đường đi ngắn nhất giữa U1 và U4 (BFS): {' -> '.join(path)} (Khoảng cách: {dist} bước)")
    
    num_comps, comps = social_net.dfs_count_connected_components()
    print(f"Số cụm cộng đồng liên thông (DFS): {num_comps} cụm")
    for i, c in enumerate(comps, 1):
        print(f"  Cụm {i}: {c}")

    # 2. Chuỗi Markov dự báo hành vi mua sắm
    print("\n--- 3.2. CHUỖI MARKOV DỰ BÁO HÀNH VI KHÁCH HÀNG ---")
    states = ['Xem_hang', 'Mua_hang', 'Roi_di']
    # Ma trận P:
    #             Xem   Mua   Rời
    # Xem hàng  [0.6,  0.3,  0.1]
    # Mua hàng  [0.4,  0.5,  0.1]
    # Rời đi    [0.2,  0.1,  0.7]
    P = [
        [0.6, 0.3, 0.1],
        [0.4, 0.5, 0.1],
        [0.2, 0.1, 0.7]
    ]
    markov = MarkovChainBuyerBehavior(states, P)
    v0 = [1.0, 0.0, 0.0]  # Ngày 0: 100% khách hàng đang ở trạng thái "Xem hàng"
    
    v_3 = markov.predict_state_distribution(v0, k_days=3)
    v_7 = markov.predict_state_distribution(v0, k_days=7)
    pi_steady = markov.compute_steady_state()
    
    print("Phân phối xác suất ban đầu (Ngày 0):", dict(zip(states, v0)))
    print("Dự báo phân phối xác suất sau 3 ngày:", dict(zip(states, v_3)))
    print("Dự báo phân phối xác suất sau 7 ngày:", dict(zip(states, v_7)))
    print("Trạng thái dừng cân bằng dài hạn (pi*):", dict(zip(states, pi_steady)))

    # 3. Kiểm tra đồ thị hai phía
    print("\n--- 3.3. KIỂM TRA ĐỒ THỊ HAI PHÍA (KHO HÀNG - CỬA HÀNG) ---")
    bipartite_graph = {
        'Kho_A': ['CuaHang_1', 'CuaHang_2'],
        'Kho_B': ['CuaHang_2', 'CuaHang_3'],
        'Kho_C': ['CuaHang_3'],
        'CuaHang_1': ['Kho_A'],
        'CuaHang_2': ['Kho_A', 'Kho_B'],
        'CuaHang_3': ['Kho_B', 'Kho_C']
    }
    is_bip, color_res = is_bipartite_graph(bipartite_graph)
    print(f"Mạng lưới phân phối có phải đồ thị hai phía không? -> {is_bip}")
    if is_bip:
        set_A = [k for k, v in color_res.items() if v == 0]
        set_B = [k for k, v in color_res.items() if v == 1]
        print(f"  Tập 1 (Kho hàng): {set_A}")
        print(f"  Tập 2 (Cửa hàng): {set_B}")

    # 4. Mở rộng DSA: Dijkstra và Kruskal
    print("\n--- PHẦN MỞ RỘNG DSA: TUYẾN GIAO HÀNG & MẠNG LƯỚI TỐI THIỂU ---")
    delivery_map = {
        'Kho_TrungTam': [('Tram_Q1', 5), ('Tram_BinhThanh', 8)],
        'Tram_Q1': [('Kho_TrungTam', 5), ('Tram_Q3', 3), ('Tram_Q7', 12)],
        'Tram_BinhThanh': [('Kho_TrungTam', 8), ('Tram_ThuDuc', 6), ('Tram_Q3', 4)],
        'Tram_Q3': [('Tram_Q1', 3), ('Tram_BinhThanh', 4), ('Tram_Q7', 7)],
        'Tram_ThuDuc': [('Tram_BinhThanh', 6), ('Tram_Q7', 15)],
        'Tram_Q7': [('Tram_Q1', 12), ('Tram_Q3', 7), ('Tram_ThuDuc', 15)]
    }
    cost, route = dijkstra_cheapest_route(delivery_map, 'Kho_TrungTam', 'Tram_Q7')
    print(f"Dijkstra - Tuyến giao hàng rẻ nhất Kho_TrungTam -> Tram_Q7: {' -> '.join(route)} (Chi phí: {cost}k)")

    mst_nodes = ['Kho_1', 'Kho_2', 'Kho_3', 'Kho_4', 'Kho_5']
    mst_edges = [
        (4, 'Kho_1', 'Kho_2'),
        (2, 'Kho_1', 'Kho_3'),
        (5, 'Kho_2', 'Kho_3'),
        (10, 'Kho_2', 'Kho_4'),
        (3, 'Kho_3', 'Kho_4'),
        (8, 'Kho_4', 'Kho_5'),
        (7, 'Kho_3', 'Kho_5')
    ]
    mst_cost, mst_tree = kruskal_minimum_spanning_tree(mst_nodes, mst_edges)
    print(f"Kruskal DSU - Cây khung nhỏ nhất liên kết kho hàng: Tổng chi phí = {mst_cost}")
    for u, v, w in mst_tree:
        print(f"  Cạnh kết nối: {u} <---> {v} (Trọng số: {w})")


if __name__ == "__main__":
    demo_module_3()

