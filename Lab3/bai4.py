"""
LAB 3 - BÀI 4: Chiếu dữ liệu lên Thành phần chính (Projection onto Principal Component)
Đề bài:
- Viết hàm project_data_1d(X_centered, pc_vector) nhận vào:
    + Ma trận dữ liệu đã chuẩn hóa tâm X_centered (M hàng, N cột)
    + Vector riêng chuẩn hóa pc_vector (độ dài vector = 1, kích thước N)
- Trả về danh sách 1D gồm M giá trị tọa độ mới tương ứng với tích vô hướng giữa mỗi hàng và pc_vector.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def project_data_1d(X_centered, pc_vector):
    """
    Chiếu dữ liệu đa chiều lên 1 trục thành phần chính (Principal Component).
    
    Tham số:
        X_centered (list of list): Ma trận M hàng x N cột.
        pc_vector (list): Vector riêng đơn vị độ dài N.
        
    Trả về:
        list: Danh sách M giá trị tọa độ 1D mới.
    """
    M = len(X_centered)
    N = len(X_centered[0])
    
    if len(pc_vector) != N:
        raise ValueError(f"Kích thước của pc_vector ({len(pc_vector)}) không khớp số cột của X ({N})!")
        
    projected = []
    for k in range(M):
        # Tích vô hướng giữa hàng k và pc_vector
        dot_product = sum(X_centered[k][j] * pc_vector[j] for j in range(N))
        projected.append(round(dot_product, 4))
        
    return projected


if __name__ == "__main__":
    # Dữ liệu chuẩn hóa tâm mẫu (M=3, N=2)
    X_centered = [
        [-1.5, -0.5],
        [ 0.0,  0.0],
        [ 1.5,  0.5]
    ]
    # Vector riêng đơn vị thứ nhất (chiều dài = sqrt(0.8^2 + 0.6^2) = 1)
    pc1 = [0.8, 0.6]

    print("--- BÀI 4: CHIẾU DỮ LIỆU LÊN THÀNH PHẦN CHÍNH ---")
    print("Dữ liệu X_centered:")
    for row in X_centered:
        print(row)
    print(f"\nVector riêng chuẩn hóa pc1: {pc1}")
    
    coords_1d = project_data_1d(X_centered, pc1)
    print(f"\nTọa độ 1D mới sau khi chiếu: {coords_1d}")

