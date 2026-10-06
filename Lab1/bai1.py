"""
LAB 1 - BÀI 1: Khởi tạo và Chuyển vị Ma trận (Transpose Matrix)
Yêu cầu:
- Cho ma trận A kích thước m x n.
- Viết hàm transpose_matrix(A) tạo ra ma trận chuyển vị A^T kích thước n x m bằng vòng lặp Python thuần.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def transpose_matrix(A):
    """
    Tính ma trận chuyển vị của ma trận A.
    
    Tham số:
        A (list of list): Ma trận đầu vào kích thước rows x cols.
        
    Trả về:
        list of list: Ma trận chuyển vị A_T kích thước cols x rows.
    """
    if not A or not A[0]:
        return []
    
    rows = len(A)
    cols = len(A[0])
    
    # Bước 2: Khởi tạo ma trận kết quả A^T có cols hàng và rows cột với giá trị 0
    A_T = [[0 for _ in range(rows)] for _ in range(cols)]
    
    # Bước 3 & 4: Hoán đổi chỉ số hàng và cột qua 2 vòng lặp lồng nhau
    for i in range(rows):
        for j in range(cols):
            A_T[j][i] = A[i][j]
            
    return A_T


if __name__ == "__main__":
    # Test case mẫu
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    
    print("--- BÀI 1: CHUYỂN VỊ MA TRẬN ---")
    print("Ma trận gốc A:")
    for row in A:
        print(row)
        
    A_T = transpose_matrix(A)
    print("\nMa trận chuyển vị A_T:")
    for row in A_T:
        print(row)
