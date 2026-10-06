"""
LAB 4 - BÀI 5: Pipeline Tối ưu Siêu tham số (Hyperparameter Grid Search) & Phân tích Đụng độ theo Dirichlet
Đề bài:
- Trong huấn luyện mô hình học máy, Grid Search duyệt qua toàn bộ không gian tổ hợp siêu tham số.
- Cho param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'optimizer': ['Adam', 'SGD']
  }
- Yêu cầu:
    1. Viết hàm custom_grid_search(param_grid) sử dụng thuật toán Quay lui để sinh toàn bộ các cấu hình tham số (Dictionary)
       mà không hardcode các vòng for lồng nhau.
    2. Phân tích Nguyên lý Nhân: Viết hàm tự động tính tổng số cấu hình từ param_grid bất kỳ.
    3. Đánh giá Đụng độ Dirichlet: Giả sử lưu 105 mô hình vào 10 cụm máy chủ theo mã băm cấu hình.
       Tính số lượng cấu hình chắc chắn xếp chung vào ít nhất 1 cụm máy chủ theo Dirichlet mở rộng ceil(N / k)
       và giải thích ý nghĩa trong cân bằng tải hệ thống AI.
"""

import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def count_total_configurations(param_grid):
    """
    Tính tổng số cấu hình siêu tham số dựa theo Nguyên lý Nhân:
    |Configurations| = Product(|Values_i|)
    """
    total = 1
    for key, values in param_grid.items():
        total *= len(values)
    return total


def custom_grid_search(param_grid):
    """
    Sinh toàn bộ các tổ hợp cấu hình tham số bằng thuật toán Quay lui (Backtracking).
    Có thể xử lý linh hoạt param_grid có số lượng tham số tùy ý mà không cần hardcode vòng lặp.
    
    Tham số:
        param_grid (dict): Từ điển ánh xạ {tên_tham_số: danh_sách_giá_trị}.
        
    Trả về:
        list of dict: Danh sách tất cả các cấu hình độc lập.
    """
    param_keys = list(param_grid.keys())
    all_configurations = []
    current_config = {}
    
    def backtrack(key_index):
        # Điều kiện dừng: Đã chọn xong giá trị cho tất cả các siêu tham số
        if key_index == len(param_keys):
            all_configurations.append(dict(current_config))
            return
            
        current_param_name = param_keys[key_index]
        candidate_values = param_grid[current_param_name]
        
        # Thử từng giá trị của tham số hiện tại
        for val in candidate_values:
            # Chọn (Assign)
            current_config[current_param_name] = val
            # Đệ quy sang tham số tiếp theo
            backtrack(key_index + 1)
            # Quay lui (Undo)
            del current_config[current_param_name]
            
    backtrack(0)
    return all_configurations


def calculate_dirichlet_collision(num_models, num_servers):
    """
    Tính số lượng cấu hình tối thiểu chắc chắn bị xếp chung vào ít nhất 1 máy chủ
    theo Nguyên lý Dirichlet mở rộng: ceil(N / k).
    """
    return math.ceil(num_models / num_servers)


# ==============================================================================
# GIẢI THÍCH Ý NGHĨA TRONG CÂN BẰNG TẢI HỆ THỐNG AI (LOAD BALANCING IN AI SYSTEMS):
#
# 1. Bản chất toán học của hiện tượng đụng độ:
#    Theo Nguyên lý Dirichlet mở rộng (Generalized Pigeonhole Principle):
#    Nếu phân phối N = 105 mô hình vào k = 10 cụm máy chủ (Server Buckets), chắc chắn tồn tại
#    ít nhất 1 cụm máy chủ phải chứa tối thiểu ceil(105 / 10) = 11 mô hình, bất kể hàm băm (Hash Function)
#    được thiết kế tốt đến mức nào.
#
# 2. Hệ quả đối với hệ thống huấn luyện và phục vụ mô hình AI (Model Serving & Storage):
#    - Điểm nghẽn tài nguyên (Hotspot Nodes): Nếu các mô hình phân bổ không đồng đều, máy chủ gánh 11 mô hình
#      (hoặc nhiều hơn) sẽ chịu tải tài nguyên (VRAM, CPU, I/O mạng) nặng hơn đáng kể so với các máy chủ khác.
#    - Nguy cơ cạn kiệt bộ nhớ (OOM - Out of Memory): Các mô hình Deep Learning có dung lượng checkpoint rất lớn
#      (vài GB đến hàng chục GB). Việc tập trung nhiều checkpoint trên một bucket có thể dẫn đến tràn ổ đĩa
#      hoặc nghẽn băng thông truy xuất.
#
# 3. Giải pháp kỹ thuật trong thực tế:
#    - Consistent Hashing (Băm nhất quán) kết hợp nút ảo (Virtual Nodes) để phân bổ đều các bucket.
#    - Dynamic Load Balancing & Autoscaling: Giám sát tải trọng thời gian thực và tự động điều phối, di dời (migrate)
#      các mô hình đang hoạt động sang các server rảnh rỗi.
# ==============================================================================

if __name__ == "__main__":
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'optimizer': ['Adam', 'SGD']
    }

    print("--- BÀI 5: PIPELINE TỐI ƯU SIÊU THAM SỐ & NGUYÊN LÝ DIRICHLET ---")
    
    # 1. Phân tích Nguyên lý Nhân
    total_expected = count_total_configurations(param_grid)
    print(f"Không gian siêu tham số: {param_grid}")
    print(f"Tổng số cấu hình theo Nguyên lý Nhân: {total_expected} (Công thức: 3 * 3 * 2 = 18)")
    
    # 2. Chạy Grid Search bằng Quay lui
    configs = custom_grid_search(param_grid)
    print(f"\nSố cấu hình sinh ra bởi Quay lui: {len(configs)}")
    print("5 cấu hình mẫu đầu tiên:")
    for i, cfg in enumerate(configs[:5], 1):
        print(f"  Config {i}: {cfg}")
    print(f"  ... và {len(configs) - 5} cấu hình tiếp theo.")
    
    # 3. Đánh giá Đụng độ Dirichlet
    N_models = 105
    k_servers = 10
    min_collision = calculate_dirichlet_collision(N_models, k_servers)
    print(f"\n--- ĐÁNH GIÁ ĐỤNG ĐỘ THEO NGUYÊN LÝ DIRICHLET MỞ RỘNG ---")
    print(f"Số mô hình N = {N_models}")
    print(f"Số cụm máy chủ k = {k_servers}")
    print(f"Số mô hình chắc chắn bị xếp chung vào ít nhất 1 cụm máy chủ: ceil({N_models}/{k_servers}) = {min_collision}")

