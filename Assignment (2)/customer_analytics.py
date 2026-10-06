"""
MÔ-ĐUN 1: TIỀN XỬ LÝ MA TRẬN & NÉN CHIỀU DỮ LIỆU PCA (Bài 1, Bài 2, Bài 3)
Hệ thống Thương mại Điện tử POLY-MART

Nội dung:
- Chức năng 1.1: Khởi tạo dữ liệu khách hàng, tính khoảng cách Manhattan (L1) và Euclidean (L2).
- Chức năng 1.2: Biến đổi Hình học & Đổi cơ sở (A * x) sang không gian đặc trưng mới.
- Chức năng 1.3: Thuật toán PCA thuần (Mean Centering -> Covariance Matrix -> Chiếu lên PC1, PC2).
"""

import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ==============================================================================
# CHỨC NĂNG 1.1: KHỞI TẠO & CHUẨN HÓA VECTOR - KHOẢNG CÁCH L1, L2
# ==============================================================================

def manhattan_distance(u, v):
    """
    Tính khoảng cách Manhattan (L1 Norm khoảng cách giữa 2 vector):
    d_L1(u, v) = sum(|u_i - v_i|)
    """
    return sum(abs(u[i] - v[i]) for i in range(len(u)))


def euclidean_distance(u, v):
    """
    Tính khoảng cách Euclidean (L2 Norm khoảng cách giữa 2 vector):
    d_L2(u, v) = sqrt(sum((u_i - v_i)^2))
    """
    return math.sqrt(sum((u[i] - v[i]) ** 2 for i in range(len(u))))


def compute_pairwise_distances(data):
    """
    Tính ma trận khoảng cách đôi một giữa các khách hàng theo chuẩn L1 và L2.
    """
    M = len(data)
    l1_matrix = [[0.0 for _ in range(M)] for _ in range(M)]
    l2_matrix = [[0.0 for _ in range(M)] for _ in range(M)]
    
    for i in range(M):
        for j in range(M):
            l1_matrix[i][j] = round(manhattan_distance(data[i], data[j]), 2)
            l2_matrix[i][j] = round(euclidean_distance(data[i], data[j]), 2)
            
    return l1_matrix, l2_matrix


def min_max_normalize(data):
    """
    Chuẩn hóa Min-Max dữ liệu về đoạn [0, 1] cho từng đặc trưng (cột):
    x_norm = (x - min) / (max - min)
    """
    M = len(data)
    N = len(data[0])
    normalized = [[0.0 for _ in range(N)] for _ in range(M)]
    
    for j in range(N):
        col_vals = [data[i][j] for i in range(M)]
        min_v = min(col_vals)
        max_v = max(col_vals)
        denom = max_v - min_v if max_v != min_v else 1.0
        for i in range(M):
            normalized[i][j] = round((data[i][j] - min_v) / denom, 4)
            
    return normalized


# ==============================================================================
# CHỨC NĂNG 1.2: BIẾN ĐỔI HÌNH HỌC & ĐỔI CƠ SỞ (A * x)
# ==============================================================================

def transform_feature_space(projection_matrix, vector_x):
    """
    Chiếu vector đặc trưng x (N chiều) sang không gian mới (K chiều) qua phép nhân ma trận A * x.
    
    Tham số:
        projection_matrix (list of list): Ma trận biến đổi A kích thước K x N.
        vector_x (list): Vector đặc trưng khách hàng kích thước N.
        
    Trả về:
        list: Vector mới kích thước K trong không gian biểu diễn mới.
    """
    K = len(projection_matrix)
    N = len(projection_matrix[0])
    if len(vector_x) != N:
        raise ValueError(f"Kích thước không tương thích: cols(A)={N} != len(x)={len(vector_x)}")
        
    res = []
    for i in range(K):
        val = sum(projection_matrix[i][j] * vector_x[j] for j in range(N))
        res.append(round(val, 4))
    return res


# ==============================================================================
# CHỨC NĂNG 1.3: THUẬT TOÁN PCA THUẦN (MEAN CENTERING -> COV -> PC1, PC2)
# ==============================================================================

def mean_centering(data):
    """
    (1) Trừ giá trị trung bình từng cột để đưa tâm dữ liệu về gốc tọa độ.
    """
    M = len(data)
    N = len(data[0])
    means = [sum(data[i][j] for i in range(M)) / M for j in range(N)]
    centered = [[data[i][j] - means[j] for j in range(N)] for i in range(M)]
    return centered, means


def compute_covariance_matrix(centered_data):
    """
    (2) Tính ma trận hiệp phương sai mẫu Cov kích thước N x N:
    Cov[i][j] = sum(X_c[k][i] * X_c[k][j] for k in range(M)) / (M - 1)
    """
    M = len(centered_data)
    N = len(centered_data[0])
    cov = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            total = sum(centered_data[k][i] * centered_data[k][j] for k in range(M))
            cov[i][j] = total / (M - 1)
    return cov


def power_iteration(cov, num_iters=100):
    """
    Thuật toán Power Iteration thuần Python để trích xuất trị riêng lớn nhất và vector riêng đơn vị.
    """
    n = len(cov)
    # Khởi tạo vector ngẫu nhiên chuẩn hóa
    v = [1.0 / math.sqrt(n)] * n
    
    for _ in range(num_iters):
        # Nhân ma trận với vector: w = cov * v
        w = [sum(cov[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x ** 2 for x in w))
        if norm < 1e-12:
            break
        v = [x / norm for x in w]
        
    # Tính trị riêng Rayleigh quotient: lambda = v^T * cov * v
    eigenvalue = sum(v[i] * sum(cov[i][j] * v[j] for j in range(n)) for i in range(n))
    return eigenvalue, v


def pca_reduce_to_2d(data):
    """
    (3) Quy trình nén dữ liệu từ 4 chiều về 2 chiều bằng PCA thuần:
    - Trừ trung bình cột
    - Tính ma trận hiệp phương sai Cov (4x4)
    - Trích xuất 2 vector riêng chính PC1, PC2 (Power Iteration + Deflation)
    - Chiếu dữ liệu lên mặt phẳng [PC1, PC2]
    """
    # 1. Trừ trung bình
    X_c, col_means = mean_centering(data)
    
    # 2. Ma trận hiệp phương sai
    cov = compute_covariance_matrix(X_c)
    n = len(cov)
    
    # 3. Tìm vector riêng thứ nhất PC1
    lambda1, pc1 = power_iteration(cov)
    
    # 4. Giảm cấp ma trận (Hotelling Deflation) để tìm PC2
    cov2 = [[cov[i][j] - lambda1 * pc1[i] * pc1[j] for j in range(n)] for i in range(n)]
    lambda2, pc2 = power_iteration(cov2)
    
    # 5. Chiếu dữ liệu lên 2 trục PC1 và PC2
    data_2d = []
    for k in range(len(data)):
        proj_1 = sum(X_c[k][j] * pc1[j] for j in range(n))
        proj_2 = sum(X_c[k][j] * pc2[j] for j in range(n))
        data_2d.append([round(proj_1, 4), round(proj_2, 4)])
        
    total_var = sum(cov[i][i] for i in range(n))
    explained_ratio = (lambda1 + lambda2) / total_var if total_var > 0 else 1.0
    
    return {
        "centered_data": X_c,
        "covariance_matrix": cov,
        "eigenvalues": [lambda1, lambda2],
        "pc1": pc1,
        "pc2": pc2,
        "data_2d": data_2d,
        "explained_variance_ratio": explained_ratio
    }


def demo_module_1():
    """Hàm chạy demo kiểm thử toàn bộ Chức năng Module 1."""
    print("=" * 70)
    print("DEMO MÔ-ĐUN 1: TIỀN XỬ LÝ MA TRẬN & NÉN CHIỀU DỮ LIỆU PCA")
    print("=" * 70)
    
    # Giả lập bộ dữ liệu hồ sơ 5 khách hàng với 4 thông số:
    # [Số đơn hàng, Tổng chi tiêu (triệu VNĐ), Điểm thưởng tích lũy, Tần suất ghé thăm (lần/tháng)]
    customer_data = [
        [15.0, 45.0, 320.0, 12.0],
        [ 3.0,  5.0,  40.0,  2.0],
        [28.0, 90.0, 650.0, 20.0],
        [ 8.0, 18.0, 150.0,  5.0],
        [22.0, 70.0, 480.0, 16.0]
    ]
    
    customer_names = ["Khách A (VIP)", "Khách B (Mới)", "Khách C (VIP Kim Cương)", "Khách D (Tiềm năng)", "Khách E (Thân thiết)"]
    
    # 1. Đo lường khoảng cách Manhattan (L1) và Euclidean (L2)
    print("\n--- 1.1. MA TRẬN KHOẢNG CÁCH GIỮA CÁC KHÁCH HÀNG ---")
    l1_mat, l2_mat = compute_pairwise_distances(customer_data)
    print("Khoảng cách Manhattan (L1) giữa Khách A và Khách B:", manhattan_distance(customer_data[0], customer_data[1]))
    print("Khoảng cách Euclidean (L2) giữa Khách A và Khách B:", round(euclidean_distance(customer_data[0], customer_data[1]), 2))
    
    # 2. Chuẩn hóa Min-Max
    print("\n--- 1.1b. DỮ LIỆU SAU CHUẨN HÓA MIN-MAX [0, 1] ---")
    norm_data = min_max_normalize(customer_data)
    for name, row in zip(customer_names, norm_data):
        print(f"  {name:25s}: {row}")
        
    # 3. Phép biến đổi hình học / chiếu đặc trưng A * x
    print("\n--- 1.2. BIẾN ĐỔI HÌNH HỌC VÀ CHIẾU ĐẶC TRƯNG (A * x) ---")
    # Giả sử ma trận trọng số chuyển đổi 2 chỉ số mới: [Chỉ số trung thành, Chỉ số tiềm năng]
    A_proj = [
        [0.1, 0.4, 0.05, 0.3],  # Trọng số tính Chỉ số Loyalty
        [0.2, 0.2, 0.01, 0.5]   # Trọng số tính Chỉ số Tiềm năng
    ]
    transformed_cust_A = transform_feature_space(A_proj, customer_data[0])
    print(f"Khách A gốc 4D: {customer_data[0]}")
    print(f"Khách A sau chiếu 2D (A * x): {transformed_cust_A}")
    
    # 4. Thuật toán PCA thuần nén 4D -> 2D
    print("\n--- 1.3. NÉN DỮ LIỆU BẰNG PCA THUẦN (4 CHIỀU -> 2 CHIỀU) ---")
    pca_res = pca_reduce_to_2d(customer_data)
    print(f"Trị riêng chính lambda1: {pca_res['eigenvalues'][0]:.2f}, lambda2: {pca_res['eigenvalues'][1]:.2f}")
    print(f"Vector riêng PC1: {[round(x, 4) for x in pca_res['pc1']]}")
    print(f"Vector riêng PC2: {[round(x, 4) for x in pca_res['pc2']]}")
    print(f"Tỷ lệ phương sai giải thích được: {pca_res['explained_variance_ratio'] * 100:.2f}%")
    print("\nTọa độ 2D của các khách hàng sau khi nén:")
    for name, coords in zip(customer_names, pca_res["data_2d"]):
        print(f"  {name:25s} -> 2D Coords [PC1, PC2]: {coords}")


if __name__ == "__main__":
    demo_module_1()

