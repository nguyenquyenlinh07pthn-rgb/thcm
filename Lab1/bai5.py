"""
LAB 1 - BÀI 5: Lập trình Thuật toán Khử Gauss Đưa Ma trận về Dạng Bậc thang (Gaussian Elimination)
Yêu cầu:
- Cho ma trận bổ sung [A|b] kích thước m x n của hệ phương trình tuyến tính.
- Thực hiện:
    + Hoán đổi dòng (Partial Pivoting): Với mỗi cột k, tìm hàng i >= k có |A[i][k]| lớn nhất, hoán đổi hàng i với hàng k.
    + Khử xuôi (Forward Elimination): Với các hàng i > k phía dưới, tính factor = A[i][k] / A[k][k], trừ hàng i đi factor * hàng k.
    + Làm tròn 2 chữ số thập phân.
    + Đánh giá độ phức tạp thuật toán O(n^3).
"""

import sys
import copy

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def gaussian_elimination(aug_matrix):
    """
    Đưa ma trận bổ sung về dạng bậc thang (Row Echelon Form) bằng thuật toán khử Gauss.
    
    Tham số:
        aug_matrix (list of list): Ma trận bổ sung [A|b] kích thước m x n.
        
    Trả về:
        list of list: Ma trận bậc thang sau khi làm tròn 2 chữ số thập phân.
    """
    # Tạo bản sao sâu để tránh thay đổi ma trận gốc
    M = copy.deepcopy(aug_matrix)
    rows = len(M)
    cols = len(M[0])
    
    # Số bước khử chính tối đa bằng min(rows, cols - 1)
    num_pivots = min(rows, cols - 1)
    
    for k in range(num_pivots):
        # 1. Hoán đổi dòng (Partial Pivoting)
        # Tìm hàng i >= k có giá trị tuyệt đối |M[i][k]| lớn nhất
        max_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, rows):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_row = i
                
        # Nếu phần tử trụ gần bằng 0 (cột không có pivot), bỏ qua cột này
        if abs(M[max_row][k]) < 1e-12:
            continue
            
        # Hoán đổi hàng max_row với hàng k
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            
        # 2. Khử xuôi (Forward Elimination)
        # Biến đổi các hàng bên dưới hàng k để đưa các phần tử cột k về 0
        pivot = M[k][k]
        for i in range(k + 1, rows):
            factor = M[i][k] / pivot
            # Cột k gán trực tiếp bằng 0 để triệt tiêu sai số dấu phẩy động
            M[i][k] = 0.0
            for j in range(k + 1, cols):
                M[i][j] -= factor * M[k][j]
                
    # 3. Làm tròn kết quả đến 2 chữ số thập phân
    result = []
    for r in range(rows):
        result_row = [round(val, 2) for val in M[r]]
        result.append(result_row)
        
    return result


# ==============================================================================
# PHÂN TÍCH ĐỘ PHỨC TẠP THỜI GIAN (TIME COMPLEXITY ANALYSIS):
#
# Giả sử ma trận vuông kích thước n x n (ma trận bổ sung n x (n + 1)):
# 1. Vòng lặp ngoài k chạy từ 0 đến n - 1 (n bước).
# 2. Tại bước k:
#    - Tìm pivot: duyệt n - k hàng -> O(n - k) phép so sánh.
#    - Khử xuôi: duyệt (n - 1 - k) hàng bên dưới.
#    - Với mỗi hàng, duyệt qua (n + 1 - k) cột để nhân và trừ:
#      Số phép tính tại bước k xấp xỉ (n - k) * (n - k) = (n - k)^2.
# 3. Tổng số phép toán số học thực hiện là:
#    Sum_{k=1}^{n} (n - k)^2 = Sum_{j=1}^{n-1} j^2 = (n - 1) * n * (2n - 1) / 6 ≈ (2/3) * n^3 / 2 = n^3 / 3.
#
# => Kết luận: Độ phức tạp thời gian của Thuật toán Khử Gauss là O(n^3).
#    Khi áp dụng trong Machine Learning (như giải Closed-form OLS Linear Regression w = (X^T X)^(-1) X^T y),
#    phép khử Gauss có độ phức tạp O(d^3) với d là số chiều đặc trưng.
# ==============================================================================

if __name__ == "__main__":
    # Test case mẫu theo đề bài:
    # 2*x + y - z = 8
    # -3*x - y + 2*z = -11
    # -2*x + y + 2*z = -3
    augmented_matrix = [
        [ 2.0,  1.0, -1.0,   8.0],
        [-3.0, -1.0,  2.0, -11.0],
        [-2.0,  1.0,  2.0,  -3.0]
    ]

    print("--- BÀI 5: THUẬT TOÁN KHỬ GAUSS (GAUSSIAN ELIMINATION) ---")
    print("Ma trận bổ sung ban đầu [A|b]:")
    for row in augmented_matrix:
        print(row)
        
    ref_matrix = gaussian_elimination(augmented_matrix)
    print("\nMa trận ở dạng bậc thang (Row Echelon Form) sau khi khử Gauss (làm tròn 2 chữ số):")
    for row in ref_matrix:
        print(row)
