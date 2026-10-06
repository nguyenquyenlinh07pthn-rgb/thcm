"""
LAB 3 - BÀI 1: Xác thực Trị riêng & Vector riêng (Eigenvalue & Eigenvector Verification)
Yêu cầu:
- Cho ma trận A = [[4, 2], [1, 3]], vector x = [2, 1], và lambda = 5.
- Viết hàm verify_eigen(A, x, lambda_val, eps=1e-6) để kiểm tra xem x có phải là vector riêng của A ứng với trị riêng lambda_val hay không (A * x == lambda * x).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def verify_eigen(A, x, lambda_val, eps=1e-6):
    """
    Kiểm tra vector x có phải là vector riêng ứng với trị riêng lambda_val của ma trận A.
    
    Tham số:
        A (list of list): Ma trận vuông cấp n.
        x (list): Vector kích thước n.
        lambda_val (float): Trị riêng cần kiểm tra.
        eps (float): Ngưỡng sai số cho phép.
        
    Trả về:
        tuple: (is_valid, Ax, lambda_x)
    """
    n = len(A)
    
    # Bước 1: Tính vế trái LHS = A * x
    Ax = []
    for i in range(n):
        Ax.append(sum(A[i][j] * x[j] for j in range(len(x))))
        
    # Bước 2: Tính vế phải RHS = lambda * x
    lambda_x = [lambda_val * val for val in x]
    
    # Bước 3: So sánh sai số abs(Ax[i] - lambda_x[i]) < eps
    is_valid = all(abs(Ax[i] - lambda_x[i]) < eps for i in range(n))
    
    return is_valid, Ax, lambda_x


if __name__ == "__main__":
    A = [
        [4, 2],
        [1, 3]
    ]
    x = [2, 1]
    lambda_val = 5

    print("--- BÀI 1: XÁC THỰC TRỊ RIÊNG & VECTOR RIÊNG ---")
    print(f"Ma trận A: {A}")
    print(f"Vector x: {x}")
    print(f"Trị riêng lambda: {lambda_val}")
    
    is_valid, Ax, lambda_x = verify_eigen(A, x, lambda_val)
    print(f"\nVế trái (A * x): {Ax}")
    print(f"Vế phải (lambda * x): {lambda_x}")
    print(f"Kết luận: x có phải là vector riêng của A không? -> {is_valid}")

