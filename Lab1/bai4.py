"""
LAB 1 - BÀI 4: Nhân hai Ma trận (C = A x B)
Yêu cầu:
- Viết hàm matrix_multiply(A, B) thực hiện phép nhân ma trận:
    + Ma trận A kích thước m x n
    + Ma trận B kích thước n x p
    + Trả về ma trận C kích thước m x p
- Yêu cầu kỹ thuật:
    + Kiểm tra điều kiện: cols(A) == rows(B). Nếu sai trả về None.
    + Dùng 3 vòng lặp for lồng nhau (không dùng numpy/thư viện ngoài).
    + Đếm và in ra tổng số phép nhân số học đã thực hiện.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def matrix_multiply(A, B):
    """
    Nhân hai ma trận A và B bằng 3 vòng lặp for lồng nhau.
    
    Tham số:
        A (list of list): Ma trận m x n
        B (list of list): Ma trận n x p
        
    Trả về:
        list of list: Ma trận C kích thước m x p, hoặc None nếu kích thước không khớp.
    """
    if not A or not A[0] or not B or not B[0]:
        print("Lỗi: Ma trận đầu vào rỗng!")
        return None
        
    m = len(A)
    n_A = len(A[0])
    n_B = len(B)
    p = len(B[0])
    
    # 1. Kiểm tra điều kiện số cột của A phải bằng số hàng của B
    if n_A != n_B:
        print(f"Lỗi: Kích thước không tương thích! Số cột A ({n_A}) != Số hàng B ({n_B})")
        return None
        
    n = n_A
    # Khởi tạo ma trận kết quả C kích thước m x p với giá trị 0
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    multiplication_count = 0  # Biến đếm số phép nhân số học
    
    # 2. Sử dụng 3 vòng lặp for lồng nhau
    for i in range(m):
        for j in range(p):
            total_sum = 0
            for k in range(n):
                total_sum += A[i][k] * B[k][j]
                multiplication_count += 1
            C[i][j] = total_sum
            
    # 3. In ra tổng số phép nhân số học đã thực hiện
    print(f"-> Tổng số phép nhân số học đã thực hiện: {multiplication_count} (Công thức: m * p * n = {m} * {p} * {n} = {m * p * n})")
    
    return C


if __name__ == "__main__":
    # Test case mẫu
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]  # Kích thước 2x3

    B = [
        [7, 8],
        [9, 1],
        [2, 3]
    ]  # Kích thước 3x2

    print("--- BÀI 4: NHÂN HAI MA TRẬN ---")
    print("Ma trận A (2x3):")
    for r in A:
        print(r)
        
    print("\nMa trận B (3x2):")
    for r in B:
        print(r)
        
    print("\nThực hiện phép nhân A x B:")
    C = matrix_multiply(A, B)
    
    print("\nMa trận kết quả C (2x2):")
    for r in C:
        print(r)
    # Kỳ vọng:
    # [[31, 19],
    #  [85, 55]]
