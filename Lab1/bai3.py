"""
LAB 1 - BÀI 3: Tích Ma trận và Vector (y = W * x)
Yêu cầu:
- Trong mạng nơ-ron truyền thẳng, đầu ra tại một tầng ẩn là y = W * x.
- Viết hàm matrix_vector_multiply(W, x) nhận vào:
    + Ma trận trọng số W (m x n)
    + Vector đầu vào x (n)
- Ràng buộc: Kiểm tra số cột của W phải bằng số phần tử của x. Nếu không hợp lệ, in thông báo lỗi và trả về None.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def matrix_vector_multiply(W, x):
    """
    Thực hiện phép nhân ma trận W với vector x (y = W * x).
    
    Tham số:
        W (list of list): Ma trận kích thước m x n.
        x (list): Vector kích thước n.
        
    Trả về:
        list: Vector y kích thước m nếu hợp lệ, ngược lại None.
    """
    if not W or not W[0]:
        print("Lỗi: Ma trận W rỗng!")
        return None
        
    m = len(W)
    n = len(W[0])
    
    # Kiểm tra ràng buộc hợp lệ: Số cột của W == số phần tử của x
    if n != len(x):
        print(f"Lỗi kích thước không tương thích: Số cột của W ({n}) khác số chiều của x ({len(x)})!")
        return None
        
    y = [0.0] * m
    for i in range(m):
        # Tích vô hướng hàng W[i] với vector x
        dot_product = sum(W[i][j] * x[j] for j in range(n))
        y[i] = round(dot_product, 6)  # Làm tròn để tránh sai số dấu phẩy động
        
    return y


if __name__ == "__main__":
    # Test case mẫu
    W = [
        [0.2, 0.5, -0.1],
        [0.8, -0.3, 0.4]
    ]
    x = [10, 2, 5]
    
    print("--- BÀI 3: NHÂN MA TRẬN VÀ VECTOR ---")
    print(f"Ma trận W: {W}")
    print(f"Vector x: {x}")
    y = matrix_vector_multiply(W, x)
    print(f"Kết quả y = W * x: {y}")  # Kỳ vọng: [2.5, 9.4]
    
    # Test case lỗi kích thước
    x_invalid = [10, 2]
    print("\nKiểm tra trường hợp kích thước không hợp lệ:")
    matrix_vector_multiply(W, x_invalid)
