"""
LAB 3 - BÀI 3: Trừ trung bình & Ma trận Hiệp phương sai (Mean Centering & Covariance Matrix)
Đề bài:
- Viết 2 hàm:
    + mean_centering(X): Nhận vào ma trận dữ liệu X (M hàng, N cột), tính trung bình từng cột và trả về ma trận đã chuẩn hóa tâm X_centered.
    + compute_covariance_matrix(X_centered): Tính ma trận hiệp phương sai Cov (N x N) theo công thức:
        Cov[i][j] = sum(X_centered[k][i] * X_centered[k][j] for k in range(M)) / (M - 1)
- Ràng buộc: Không dùng thư viện numpy. Xử lý đúng với kích thước ma trận bất kỳ.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def mean_centering(X):
    """
    Chuẩn hóa dữ liệu về tâm (trừ giá trị trung bình từng cột).
    
    Tham số:
        X (list of list): Ma trận dữ liệu kích thước M hàng x N cột.
        
    Trả về:
        list of list: Ma trận X_centered kích thước M x N.
    """
    if not X or not X[0]:
        return []
        
    M = len(X)
    N = len(X[0])
    
    # Tính giá trị trung bình cho từng cột j
    col_means = [sum(X[k][j] for k in range(M)) / M for j in range(N)]
    
    # Trừ giá trị trung bình tương ứng
    X_centered = []
    for k in range(M):
        row_centered = [X[k][j] - col_means[j] for j in range(N)]
        X_centered.append(row_centered)
        
    return X_centered


def compute_covariance_matrix(X_centered):
    """
    Tính ma trận hiệp phương sai mẫu (Sample Covariance Matrix) kích thước N x N.
    
    Tham số:
        X_centered (list of list): Ma trận dữ liệu đã chuẩn hóa tâm (M hàng x N cột).
        
    Trả về:
        list of list: Ma trận hiệp phương sai Cov (N x N).
    """
    M = len(X_centered)
    N = len(X_centered[0])
    
    if M <= 1:
        raise ValueError("Số lượng mẫu M phải lớn hơn 1 để tính phương sai mẫu (chia cho M - 1)!")
        
    Cov = [[0.0 for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            total = sum(X_centered[k][i] * X_centered[k][j] for k in range(M))
            Cov[i][j] = total / (M - 1)
            
    return Cov


if __name__ == "__main__":
    # Test case mẫu: Dữ liệu điểm thi của 4 sinh viên qua 3 môn
    X_sample = [
        [8.0, 7.0, 9.0],
        [6.0, 5.0, 6.0],
        [9.0, 8.0, 8.0],
        [7.0, 6.0, 7.0]
    ]

    print("--- BÀI 3: TRỪ TRUNG BÌNH & MA TRẬN HIỆP PHƯƠNG SAI ---")
    print(f"Dữ liệu gốc X (M={len(X_sample)}, N={len(X_sample[0])}):")
    for row in X_sample:
        print(row)
        
    X_c = mean_centering(X_sample)
    print("\nMa trận sau khi trừ trung bình (X_centered):")
    for row in X_c:
        print([round(val, 4) for val in row])
        
    cov_matrix = compute_covariance_matrix(X_c)
    print("\nMa trận hiệp phương sai mẫu (Covariance Matrix N x N):")
    for row in cov_matrix:
        print([round(val, 4) for val in row])

