"""
LAB 3 - BÀI 5: Pipeline Giảm chiều Dữ liệu Bệnh án & Đánh giá Tỷ lệ Thông tin giữ lại (Explained Variance)
Đề bài:
- Bộ dữ liệu y tế gồm 4 thông số: [Huyết áp, Đường huyết, Cholesterol, BMI]. Nén từ 4 chiều về 2 chiều để trực quan hóa.
- Yêu cầu:
    1. Viết hàm pca_reduce_2d(data): tự động trừ trung bình, tính ma trận hiệp phương sai 4x4, và chiếu dữ liệu lên 2 trục vector riêng PC1, PC2.
    2. Tính Tỷ lệ phương sai giải thích được: Với lambda = [145.2, 32.8, 4.5, 1.2], tính:
       Ratio = (lambda_1 + lambda_2) / sum(lambda).
    3. Phân tích ý nghĩa học máy: Viết comment phân tích ý nghĩa bảo toàn thông tin và lợi ích trực quan hóa 2D cho y tế.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_mean_centered(data):
    """Trừ giá trị trung bình từng cột."""
    M = len(data)
    N = len(data[0])
    means = [sum(data[i][j] for i in range(M)) / M for j in range(N)]
    return [[data[i][j] - means[j] for j in range(N)] for i in range(M)], means


def get_covariance(X_centered):
    """Tính ma trận hiệp phương sai mẫu N x N."""
    M = len(X_centered)
    N = len(X_centered[0])
    cov = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            cov[i][j] = sum(X_centered[k][i] * X_centered[k][j] for k in range(M)) / (M - 1)
    return cov


def power_iteration(matrix, num_iters=100):
    """
    Thuật toán Power Iteration thuần Python để tìm trị riêng trội nhất và vector riêng tương ứng.
    """
    n = len(matrix)
    # Khởi tạo vector đơn vị ban đầu
    v = [1.0 / (n ** 0.5)] * n
    
    for _ in range(num_iters):
        # Nhân ma trận với vector: w = matrix * v
        w = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = sum(x ** 2 for x in w) ** 0.5
        if norm < 1e-12:
            break
        v = [x / norm for x in w]
        
    # Tính trị riêng tương ứng: lambda = v^T * matrix * v
    eigenvalue = sum(v[i] * sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n))
    return eigenvalue, v


def pca_reduce_2d(data):
    """
    Pipeline PCA thu gọn:
    - Trừ trung bình dữ liệu
    - Tính ma trận hiệp phương sai 4x4
    - Tìm 2 vector riêng PC1, PC2
    - Chiếu dữ liệu đa chiều về không gian 2D
    
    Tham số:
        data (list of list): Bộ dữ liệu đầu vào kích thước M x 4.
        
    Trả về:
        tuple: (data_2d, pc1, pc2, cov_matrix)
    """
    # 1. Trừ trung bình
    X_c, col_means = get_mean_centered(data)
    
    # 2. Ma trận hiệp phương sai 4x4
    cov = get_covariance(X_c)
    n = len(cov)
    
    # 3. Tìm vector riêng thứ nhất (PC1)
    val1, pc1 = power_iteration(cov)
    
    # 4. Giảm cấp ma trận (Hotelling Deflation) để tìm PC2
    cov2 = [
        [cov[i][j] - val1 * pc1[i] * pc1[j] for j in range(n)]
        for i in range(n)
    ]
    val2, pc2 = power_iteration(cov2)
    
    # 5. Chiếu dữ liệu lên 2 trục PC1 và PC2
    data_2d = []
    for k in range(len(data)):
        proj_pc1 = sum(X_c[k][j] * pc1[j] for j in range(n))
        proj_pc2 = sum(X_c[k][j] * pc2[j] for j in range(n))
        data_2d.append([round(proj_pc1, 4), round(proj_pc2, 4)])
        
    return data_2d, pc1, pc2, cov


# ==============================================================================
# PHÂN TÍCH Ý NGHĨA HỌC MÁY (MACHINE LEARNING INTERPRETATION):
#
# 1. Nếu tỷ lệ thông tin giữ lại đạt trên 90% (ở đây đạt xấp xỉ 96.9%), việc loại bỏ 2 chiều cuối
#    KHÔNG làm mất bản chất dữ liệu:
#    - Tỷ lệ phương sai giải thích được (Explained Variance Ratio) phản ánh mức độ bảo toàn độ phân tán
#      (sự biến thiên thông tin phân biệt giữa các bệnh nhân).
#    - Khi 2 thành phần chính PC1 và PC2 đã nắm giữ tới hơn 96.9% phương sai, nghĩa là 2 chiều còn lại chỉ
#      đóng góp dưới 3.1% thông tin. Phần này phần lớn là nhiễu đo lường ngẫu nhiên (noise) hoặc các yếu tố
#      trùng lặp (multicollinearity) giữa các chỉ số sinh hóa.
#    - Việc loại bỏ 2 chiều cuối giúp giảm chiều dữ liệu (Dimensionality Reduction), hạn chế hiện tượng
#      "Lời nguyền số chiều" (Curse of Dimensionality), khử nhiễu và giúp mô hình học máy tránh Overfitting.
#
# 2. Lợi ích khi trực quan hóa 2D cho chuyên gia y tế:
#    - Khả năng tiếp cận trực quan: Não người và màn hình hiển thị chỉ có thể quan sát trực tiếp tốt nhất ở không gian 2D/3D.
#      Không gian 4 chiều y tế [Huyết áp, Đường huyết, Cholesterol, BMI] là bất khả thi để vẽ trên một biểu đồ phân tán duy nhất.
#    - Phân cụm và phát hiện bất thường (Clustering & Anomaly Detection): Bác sĩ có thể nhận diện ngay các cụm bệnh nhân có
#      nguy cơ tim mạch/tiểu đường tương đồng, phân biệt nhóm nguy cơ cao với nhóm bình thường hoặc phát hiện các ca bệnh dị biệt (Outliers).
#    - Hỗ trợ ra quyết định lâm sàng nhanh chóng: Giúp đơn giản hóa bức tranh toàn cảnh về sức khỏe của bệnh nhân mà không bị phân tâm
#      bởi bảng số liệu rối rắm nhiều cột.
# ==============================================================================

if __name__ == "__main__":
    medical_data = [
        [120, 95, 210, 24.5],
        [140, 130, 250, 29.0],
        [110, 85, 180, 21.5],
        [155, 160, 280, 32.0],
        [130, 105, 220, 26.0]
    ]

    print("--- BÀI 5: PIPELINE GIẢM CHIỀU DỮ LIỆU BỆNH ÁN (PCA 2D) ---")
    data_2d, pc1, pc2, cov_mat = pca_reduce_2d(medical_data)
    
    print("Ma trận hiệp phương sai 4x4:")
    for row in cov_mat:
        print([round(val, 2) for val in row])
        
    print(f"\nVector riêng PC1: {[round(x, 4) for x in pc1]}")
    print(f"Vector riêng PC2: {[round(x, 4) for x in pc2]}")
    
    print("\nTọa độ 5 bệnh nhân sau khi giảm về 2D [PC1, PC2]:")
    for i, pt in enumerate(data_2d, 1):
        print(f"  Bệnh nhân {i}: {pt}")
        
    # Tính Tỷ lệ phương sai giải thích được theo đề bài
    lambdas = [145.2, 32.8, 4.5, 1.2]
    ratio = (lambdas[0] + lambdas[1]) / sum(lambdas)
    print(f"\n--- ĐÁNH GIÁ TỶ LỆ THÔNG TIN GIỮ LẠI (EXPLAINED VARIANCE) ---")
    print(f"Các trị riêng lambda: {lambdas}")
    print(f"Tổng phương sai: {sum(lambdas):.2f}")
    print(f"Phương sai giữ lại bởi 2 chiều đầu (PC1 + PC2): {lambdas[0] + lambdas[1]:.2f}")
    print(f"Tỷ lệ thông tin giữ lại (Ratio): {ratio * 100:.2f}%")

