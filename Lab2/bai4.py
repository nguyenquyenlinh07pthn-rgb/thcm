"""
LAB 2 - BÀI 4: Phép biến đổi Hình học (Co giãn và Xoay)
Đề bài:
- Viết các hàm:
    + scale_points(points, sx, sy)
    + rotate_points(points, angle_degrees)
  áp dụng các ma trận biến đổi hình học trên tập điểm 2D.
- Yêu cầu kỹ thuật:
    + Định nghĩa ma trận xoay R = [[cos(rad), -sin(rad)], [sin(rad), cos(rad)]] với rad = math.radians(angle_degrees).
    + Làm tròn kết quả đến 2 chữ số thập phân.
"""

import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def scale_points(points, sx, sy):
    """
    Áp dụng phép biến đổi co giãn (Scaling) lên danh sách các điểm 2D.
    Ma trận co giãn S:
        [sx,  0]
        [ 0, sy]
    """
    transformed = []
    for p in points:
        x, y = p[0], p[1]
        x_new = round(x * sx, 2)
        y_new = round(y * sy, 2)
        transformed.append([x_new, y_new])
    return transformed


def rotate_points(points, angle_degrees):
    """
    Áp dụng phép biến đổi xoay (Rotation) theo góc angle_degrees ngược chiều kim đồng hồ quanh gốc tọa độ.
    Ma trận xoay R:
        [cos(rad), -sin(rad)]
        [sin(rad),  cos(rad)]
    """
    rad = math.radians(angle_degrees)
    cos_rad = math.cos(rad)
    sin_rad = math.sin(rad)
    
    transformed = []
    for p in points:
        x, y = p[0], p[1]
        # x' = x * cos(rad) - y * sin(rad)
        # y' = x * sin(rad) + y * cos(rad)
        x_new = round(x * cos_rad - y * sin_rad, 2)
        y_new = round(x * sin_rad + y * cos_rad, 2)
        transformed.append([x_new, y_new])
    return transformed


if __name__ == "__main__":
    # Danh sách các điểm 2D mẫu (ví dụ các đỉnh của một hình chữ nhật)
    sample_points = [
        [1.0, 1.0],
        [4.0, 1.0],
        [4.0, 3.0],
        [1.0, 3.0]
    ]

    print("--- BÀI 4: PHÉP BIẾN ĐỔI HÌNH HỌC (CO GIÃN & XOAY) ---")
    print(f"Các điểm ban đầu: {sample_points}")
    
    # 1. Thử nghiệm co giãn (sx = 2.0, sy = 0.5)
    scaled = scale_points(sample_points, sx=2.0, sy=0.5)
    print(f"\nSau khi co giãn (sx=2.0, sy=0.5):")
    print(scaled)
    
    # 2. Thử nghiệm xoay (90 độ)
    rotated_90 = rotate_points(sample_points, angle_degrees=90)
    print(f"\nSau khi xoay 90 độ:")
    print(rotated_90)

