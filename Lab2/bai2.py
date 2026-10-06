"""
LAB 2 - BÀI 2: Kiểm tra Độc lập tuyến tính trong không gian 2D
Yêu cầu:
- Trong tiền xử lý dữ liệu, nếu 2 thuộc tính phụ thuộc tuyến tính (cùng phương), ta có thể loại bỏ 1 thuộc tính.
- Viết hàm is_linearly_dependent_2d(v1, v2) kiểm tra tính độc lập/phụ thuộc tuyến tính qua định thức:
    det = v1[0] * v2[1] - v1[1] * v2[0]
- Nếu abs(det) < 1e-9 -> Trả về True (phụ thuộc), ngược lại False (độc lập).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def is_linearly_dependent_2d(v1, v2):
    """
    Kiểm tra 2 vector trong không gian 2D có phụ thuộc tuyến tính hay không bằng định thức det.
    
    Tham số:
        v1 (list): Vector thứ nhất [x1, y1]
        v2 (list): Vector thứ hai [x2, y2]
        
    Trả về:
        bool: True nếu phụ thuộc tuyến tính (det == 0), False nếu độc lập tuyến tính.
    """
    det = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(det) < 1e-9:
        return True
    return False


if __name__ == "__main__":
    print("--- BÀI 2: KIỂM TRA ĐỘC LẬP TUYẾN TÍNH 2D ---")
    pair1 = ([2, 4], [4, 8])
    pair2 = ([2, 4], [1, 5])
    
    res1 = is_linearly_dependent_2d(pair1[0], pair1[1])
    res2 = is_linearly_dependent_2d(pair2[0], pair2[1])
    
    print(f"Vector {pair1[0]} và {pair1[1]}: Phụ thuộc tuyến tính? -> {res1}")  # True
    print(f"Vector {pair2[0]} và {pair2[1]}: Phụ thuộc tuyến tính? -> {res2}")  # False

