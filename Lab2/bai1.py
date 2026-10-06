"""
LAB 2 - BÀI 1: Tính Tổ hợp Tuyến tính và Tọa độ theo Cơ sở mới
Yêu cầu:
- Cho cơ sở B gồm 2 vector b1 = [1, 0] và b2 = [1, 1], vector hệ số tọa độ c = [-2, 7].
- Viết hàm compute_linear_combination(B, c) để tìm vector v = c1*b1 + c2*b2.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def compute_linear_combination(B, c):
    """
    Tính vector biểu diễn tổ hợp tuyến tính v = sum(c[i] * B[i]).
    
    Tham số:
        B (list of list): Danh sách các vector cơ sở, mỗi vector có dim phần tử.
        c (list): Vector hệ số tọa độ (len(c) == len(B)).
        
    Trả về:
        list: Vector kết quả v có số chiều bằng dim.
    """
    dim = len(B[0])
    v = [0.0] * dim
    
    for i in range(len(B)):
        for j in range(dim):
            v[j] += c[i] * B[i][j]
            
    return v


if __name__ == "__main__":
    # Dữ liệu đầu vào mẫu
    B = [
        [1, 0],  # b1
        [1, 1]   # b2
    ]
    c = [-2, 7]

    print("--- BÀI 1: TÍNH TỔ HỢP TUYẾN TÍNH ---")
    print(f"Cơ sở B: {B}")
    print(f"Hệ số tọa độ c: {c}")
    v = compute_linear_combination(B, c)
    print(f"Vector kết quả v: {v}")  # Kỳ vọng: [5.0, 7.0]

