"""
LAB 3 - BÀI 2: Tính Lũy thừa nhanh Ma trận bằng Chéo hóa (Matrix Power via Diagonalization)
Yêu cầu:
- Trong các mô hình AI chuỗi (Markov Chains, RNN), ta cần tính A^k với k lớn.
- Nếu A chéo hóa được: A = P * D * P^(-1), thì:
    A^k = P * (D^k) * P^(-1)
- Cho ma trận đường chéo D có các phần tử trên đường chéo d = [5, 2], ma trận P và P_inv.
- Viết hàm matrix_power_fast(P, D_diag, P_inv, k) để tính nhanh A^k.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def mat_mul_2x2(A, B):
    """Nhân hai ma trận 2x2 bằng Python thuần."""
    return [
        [
            A[0][0] * B[0][0] + A[0][1] * B[1][0],
            A[0][0] * B[0][1] + A[0][1] * B[1][1]
        ],
        [
            A[1][0] * B[0][0] + A[1][1] * B[1][0],
            A[1][0] * B[0][1] + A[1][1] * B[1][1]
        ]
    ]


def matrix_power_fast(P, D_diag, P_inv, k):
    """
    Tính nhanh lũy thừa ma trận A^k thông qua phép chéo hóa: A^k = P * (D^k) * P_inv.
    
    Tham số:
        P (list of list): Ma trận làm chéo hóa (chứa các vector riêng).
        D_diag (list): Danh sách các trị riêng trên đường chéo của D.
        P_inv (list of list): Ma trận nghịch đảo của P.
        k (int): Số mũ lũy thừa.
        
    Trả về:
        list of list: Ma trận A^k kích thước 2x2 (làm tròn 4 chữ số).
    """
    # Bước 1 & 2: Lũy thừa ma trận đường chéo D^k
    # D^k có dạng:
    # [[D_diag[0]**k, 0.0],
    #  [0.0, D_diag[1]**k]]
    D_k = [
        [float(D_diag[0] ** k), 0.0],
        [0.0, float(D_diag[1] ** k)]
    ]
    
    # Bước 3: Nhân 3 ma trận: P * D_k * P_inv
    P_D_k = mat_mul_2x2(P, D_k)
    A_k = mat_mul_2x2(P_D_k, P_inv)
    
    # Làm tròn để khử sai số số học
    A_k_rounded = [[round(val, 4) for val in row] for row in A_k]
    return A_k_rounded


if __name__ == "__main__":
    # Ví dụ ma trận A = [[4, 2], [1, 3]] có:
    # - Trị riêng: lambda1 = 5, lambda2 = 2 => D_diag = [5, 2]
    # - Vector riêng tương ứng: v1 = [2, 1]^T, v2 = [-1, 1]^T
    # - Ma trận P = [[2, -1], [1, 1]]
    # - Định thức det(P) = 2*1 - (-1)*1 = 3
    # - Nghịch đảo P_inv = [[1/3, 1/3], [-1/3, 2/3]]
    D_diag = [5, 2]
    P = [
        [2.0, -1.0],
        [1.0,  1.0]
    ]
    P_inv = [
        [ 1.0 / 3.0, 1.0 / 3.0],
        [-1.0 / 3.0, 2.0 / 3.0]
    ]
    
    k = 4
    print("--- BÀI 2: TÍNH LŨY THỪA NHANH MA TRẬN BẰNG CHÉO HÓA ---")
    print(f"Đường chéo D_diag = {D_diag}")
    print(f"Số mũ k = {k}")
    
    A_k = matrix_power_fast(P, D_diag, P_inv, k)
    print(f"\nKết quả A^{k}:")
    for row in A_k:
        print(row)
        
    # Kiểm tra với phép nhân liên tiếp A * A * A * A
    A = [[4.0, 2.0], [1.0, 3.0]]
    A_mul = A
    for _ in range(k - 1):
        A_mul = mat_mul_2x2(A_mul, A)
    print(f"\nKết quả nhân liên tiếp để kiểm chứng:")
    for row in A_mul:
        print([round(val, 4) for val in row])

