"""
LAB 2 - BÀI 3: Lập trình Tìm Hạt nhân Ker(f) và Số chiều Nullity qua Hệ thuần nhất A * x = 0
Đề bài:
- Hạt nhân Ker(f) của ánh xạ tuyến tính f: R^3 -> R^2 biểu diễn bởi ma trận A (2 x 3) là tập các vector x in R^3 thỏa mãn A * x = 0.
- Các bước thực hiện:
    + Đưa về RREF: Áp dụng thuật toán biến đổi dòng sơ cấp (Gauss-Jordan) đưa ma trận A về dạng:
        RREF(A) = [
            [1, 0, c1],
            [0, 1, c2]
        ]
    + Xác định vector cơ sở: Đặt biến tự do x3 = 1.0 => x1 = -c1, x2 = -c2.
    + Cài đặt hàm find_kernel_basis_2x3(A): Trả về vector cơ sở của Ker(f) và số chiều dim(Ker).
"""

import sys
import copy

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def rref_2x3(A):
    """
    Đưa ma trận 2x3 về dạng bậc thang rút gọn (Reduced Row Echelon Form - RREF).
    """
    M = [row[:] for row in A]
    
    # Bước 1: Pivot cho cột 0
    if abs(M[0][0]) < abs(M[1][0]):
        M[0], M[1] = M[1], M[0]
        
    pivot0 = M[0][0]
    if abs(pivot0) > 1e-9:
        M[0] = [val / pivot0 for val in M[0]]
        factor = M[1][0]
        M[1] = [M[1][j] - factor * M[0][j] for j in range(3)]
        
    # Bước 2: Pivot cho cột 1
    pivot1 = M[1][1]
    if abs(pivot1) > 1e-9:
        M[1] = [val / pivot1 for val in M[1]]
        factor = M[0][1]
        M[0] = [M[0][j] - factor * M[1][j] for j in range(3)]
        
    return M


def find_kernel_basis_2x3(A):
    """
    Tìm vector cơ sở của Ker(f) và số chiều Nullity (dim(Ker)) cho ánh xạ R^3 -> R^2.
    
    Tham số:
        A (list of list): Ma trận 2x3 biểu diễn ánh xạ f.
        
    Trả về:
        tuple: (kernel_basis, dim_ker, rref_matrix)
    """
    M_rref = rref_2x3(A)
    
    # RREF(A) có dạng:
    # [1, 0, c1]
    # [0, 1, c2]
    c1 = M_rref[0][2]
    c2 = M_rref[1][2]
    
    # Chọn biến tự do x3 = 1.0 => x1 = -c1, x2 = -c2
    x3 = 1.0
    x1 = -c1
    x2 = -c2
    
    basis_vector = [round(x1, 4), round(x2, 4), round(x3, 4)]
    dim_ker = 1  # Theo Định lý Hạng - Bậc (Rank-Nullity Theorem): dim(Ker) = 3 - rank(A) = 3 - 2 = 1
    
    return basis_vector, dim_ker, M_rref


if __name__ == "__main__":
    # Test case mẫu: Ánh xạ f biểu diễn bởi ma trận A (2x3)
    # Ví dụ:
    # 1*x1 + 2*x2 - 1*x3 = 0
    # 2*x1 + 5*x2 + 1*x3 = 0
    A = [
        [1.0, 2.0, -1.0],
        [2.0, 5.0,  1.0]
    ]

    print("--- BÀI 3: TÌM HẠT NHÂN KER(f) VÀ NULLITY (A * x = 0) ---")
    print("Ma trận A (2x3):")
    for row in A:
        print(row)
        
    basis, dim_k, rref_m = find_kernel_basis_2x3(A)
    
    print("\nMa trận RREF(A):")
    for row in rref_m:
        print([round(val, 4) for val in row])
        
    print(f"\nVector cơ sở của Ker(f): v = {basis}")
    print(f"Số chiều Hạt nhân dim(Ker(f)) = {dim_k}")
    
    # Kiểm tra lại điều kiện A * basis ≈ 0
    check_Ax = [
        round(sum(A[i][j] * basis[j] for j in range(3)), 4)
        for i in range(2)
    ]
    print(f"Kiểm chứng A * v = {check_Ax} (xấp xỉ [0, 0])")

