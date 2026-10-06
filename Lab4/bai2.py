"""
LAB 4 - BÀI 2: Thuật toán Quay lui Sinh tất cả Tập con Đặc trưng (Feature Subsets via Backtracking)
Yêu cầu:
- Cho danh sách các tên đặc trưng: features = ['Age', 'Income', 'Score'].
- Viết hàm generate_subsets_backtracking(features) sử dụng thuật toán Quay lui (Backtracking)
  để tìm tất cả các tập con đặc trưng (bao gồm cả tập rỗng []).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_subsets_backtracking(features):
    """
    Sinh tất cả các tập con (Power Set) của danh sách features bằng thuật toán Quay lui.
    
    Tham số:
        features (list): Danh sách các phần tử/đặc trưng.
        
    Trả về:
        list of list: Danh sách tất cả 2^N tập con.
    """
    all_subsets = []
    
    def backtrack(start_index, current_path):
        # Bước 1: Lưu tập con hiện tại (tạo bản sao để tránh bị thay đổi)
        all_subsets.append(list(current_path))
        
        # Bước 2: Duyệt qua các đặc trưng tiếp theo từ vị trí start_index
        for i in range(start_index, len(features)):
            # Chọn đặc trưng
            current_path.append(features[i])
            # Gọi đệ quy bước tiếp theo
            backtrack(i + 1, current_path)
            # Quay lui (Undo lựa chọn)
            current_path.pop()
            
    # Bắt đầu quay lui từ vị trí 0 với tập rỗng
    backtrack(0, [])
    return all_subsets


if __name__ == "__main__":
    features = ['Age', 'Income', 'Score']
    print("--- BÀI 2: QUAY LUI SINH TẬP CON ĐẶC TRƯNG ---")
    print(f"Tập đặc trưng ban đầu: {features}")
    subsets = generate_subsets_backtracking(features)
    print(f"\nTổng số tập con sinh được (2^{len(features)} = {2**len(features)}): {len(subsets)}")
    print("Danh sách các tập con:")
    for idx, s in enumerate(subsets, 1):
        print(f"  {idx}: {s}")

