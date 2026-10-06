"""
LAB 2 - BÀI 5: Pipeline Tăng cường Dữ liệu Ảnh (Data Augmentation) với Tọa độ Đồng nhất
Đề bài:
- Áp dụng chuỗi biến đổi hình học (Affine Transformation): Co giãn -> Xoay -> Tịnh tiến trên Bounding Box bằng Tọa độ đồng nhất 3x3.
- Yêu cầu:
    + Viết hàm create_affine_matrix(sx, sy, angle_deg, tx, ty) tạo ma trận biến đổi 3x3.
    + Viết hàm transform_bounding_box(bbox, affine_matrix) chuyển tọa độ bbox [x, y] sang [x, y, 1], nhân ma trận và trả về tọa độ 2D mới.
    + Viết comment giải thích vì sao tọa độ đồng nhất 3x3 giúp GPU xử lý song song nhanh hơn trong Deep Learning.
"""

import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def mat_mul_3x3(A, B):
    """Hàm phụ trợ nhân hai ma trận 3x3."""
    res = [[0.0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            res[i][j] = sum(A[i][k] * B[k][j] for k in range(3))
    return res


def create_affine_matrix(sx, sy, angle_deg, tx, ty):
    """
    Tạo ma trận biến đổi Affine 3x3 kết hợp: Co giãn (S) -> Xoay (R) -> Tịnh tiến (T).
    M = T * R * S
    
    Trong tọa độ đồng nhất:
    S = [[sx, 0,  0],
         [0,  sy, 0],
         [0,  0,  1]]
         
    R = [[cos(rad), -sin(rad), 0],
         [sin(rad),  cos(rad), 0],
         [0,         0,        1]]
         
    T = [[1, 0, tx],
         [0, 1, ty],
         [0, 0, 1 ]]
    """
    rad = math.radians(angle_deg)
    cos_val = math.cos(rad)
    sin_val = math.sin(rad)
    
    # Ma trận Co giãn S
    S = [
        [sx, 0.0, 0.0],
        [0.0, sy, 0.0],
        [0.0, 0.0, 1.0]
    ]
    
    # Ma trận Xoay R
    R = [
        [cos_val, -sin_val, 0.0],
        [sin_val,  cos_val, 0.0],
        [0.0,      0.0,     1.0]
    ]
    
    # Ma trận Tịnh tiến T
    T = [
        [1.0, 0.0, float(tx)],
        [0.0, 1.0, float(ty)],
        [0.0, 0.0, 1.0]
    ]
    
    # Chuỗi biến đổi: Áp dụng S trước, rồi R, rồi T => M = T * (R * S)
    RS = mat_mul_3x3(R, S)
    affine_matrix = mat_mul_3x3(T, RS)
    
    return affine_matrix


def transform_bounding_box(bbox, affine_matrix):
    """
    Chuyển tọa độ bbox [x, y] sang tọa độ đồng nhất [x, y, 1],
    nhân với ma trận affine 3x3 và trả về tọa độ 2D mới [x', y'].
    
    Tham số:
        bbox (list): Có thể là 1 điểm [x, y] hoặc danh sách các điểm [[x1, y1], [x2, y2], ...].
        affine_matrix (list of list): Ma trận affine 3x3.
        
    Trả về:
        list: Tọa độ 2D mới đã làm tròn 2 chữ số thập phân.
    """
    # Nếu truyền vào là 1 điểm đơn [x, y]
    if isinstance(bbox[0], (int, float)):
        points = [bbox]
        is_single = True
    else:
        points = bbox
        is_single = False
        
    transformed_points = []
    for p in points:
        x, y = p[0], p[1]
        v_homogeneous = [x, y, 1.0]
        
        # Nhân ma trận affine 3x3 với vector [x, y, 1]^T
        x_new = sum(affine_matrix[0][j] * v_homogeneous[j] for j in range(3))
        y_new = sum(affine_matrix[1][j] * v_homogeneous[j] for j in range(3))
        
        transformed_points.append([round(x_new, 2), round(y_new, 2)])
        
    return transformed_points[0] if is_single else transformed_points


# ==============================================================================
# GIẢI THÍCH: VÌ SAO TỌA ĐỘ ĐỒNG NHẤT 3x3 GIÚP GPU XỬ LÝ SONG SONG NHANH HƠN?
#
# 1. Hợp nhất các phép toán (Composition of Transformations):
#    Trong không gian 2D Euclid thông thường, phép co giãn và xoay là phép nhân ma trận 
#    (tuyến tính: v' = M * v), nhưng phép tịnh tiến lại là phép cộng vector (phi tuyến: v' = v + t).
#    Điều này khiến chuỗi biến đổi phải trải qua nhiều bước cộng/nhân rời rạc: v' = R * (S * v) + t.
#    Khi nâng lên Tọa độ đồng nhất (Homogeneous Coordinates) kích thước 3x3, TẤT CẢ các phép toán 
#    (co giãn, xoay, lật, trượt, tịnh tiến) đều được biểu diễn thành MỘT phép nhân ma trận duy nhất (v' = M * v).
#
# 2. Tối ưu hóa tính toán song song trên phần cứng GPU:
#    GPU có kiến trúc phần cứng chuyên biệt (như Tensor Cores / SIMD / CUDA cores) được thiết kế để
#    thực thi cực nhanh các phép toán nhân ma trận lớn (GEMM - General Matrix Multiply).
#    Thay vì phải duyệt tuần tự và rẽ nhánh từng phép biến đổi cho hàng ngàn Bounding Box / hàng triệu điểm ảnh,
#    hệ thống có thể:
#      - Nhân trước toàn bộ chuỗi n phép biến đổi thành DUY NHẤT một ma trận kết hợp M_combined 3x3 (O(1)).
#      - Gom toàn bộ tọa độ bbox vào một Batch Tensor lớn và nhân song song với M_combined chỉ trong MỘT chu kỳ lệnh.
#    Nhờ đó, tốc độ xử lý Data Augmentation thời gian thực (Real-time Pipeline) tăng lên gấp nhiều lần.
# ==============================================================================

if __name__ == "__main__":
    # Test case mẫu: Bounding Box của một vật thể trong bài toán Object Detection
    # Giả sử bounding box được biểu diễn bằng 4 đỉnh: [top-left, top-right, bottom-right, bottom-left]
    bbox_corners = [
        [10.0, 10.0],
        [50.0, 10.0],
        [50.0, 40.0],
        [10.0, 40.0]
    ]
    
    print("--- BÀI 5: DATA AUGMENTATION VỚI TỌA ĐỘ ĐỒNG NHẤT 3x3 ---")
    print(f"Các đỉnh Bounding Box ban đầu: {bbox_corners}")
    
    # Tạo ma trận Affine: Phóng to (sx=1.5, sy=1.5), Xoay 30 độ, Tịnh tiến (tx=10, ty=20)
    affine_mat = create_affine_matrix(sx=1.5, sy=1.5, angle_deg=30, tx=10.0, ty=20.0)
    print("\nMa trận Affine 3x3:")
    for row in affine_mat:
        print([round(val, 4) for val in row])
        
    transformed_bbox = transform_bounding_box(bbox_corners, affine_mat)
    print("\nCác đỉnh Bounding Box sau biến đổi Affine:")
    print(transformed_bbox)

