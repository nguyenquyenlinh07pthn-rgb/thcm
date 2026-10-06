"""
MÔ-ĐUN 4: TỐI ƯU HÓA CHI PHÍ MARKETING & GRADIENT DESCENT (Bài 8)
Hệ thống Thương mại Điện tử POLY-MART

Nội dung:
- Chức năng 4.1: Quy hoạch Tuyến tính Hình học 2D (Geometric Linear Programming) tối đa hóa tiếp cận.
- Chức năng 4.2: Đưa về Dạng chính tắc (Slack variables s1, s2) & Thiết lập bài toán Đối ngẫu (Dual).
- Chức năng 4.3: Thuật toán Tối ưu hóa Gradient Descent 1D cực tiểu hóa hàm mất mát chi phí L(w).
- Mở rộng DSA:
    + Quy hoạch động (Knapsack 0/1) tối ưu hóa combo sản phẩm khuyến mãi.
    + Rolling Hash (Rabin-Karp) tra cứu mã voucher giảm giá tốc độ cao.
"""

import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ==============================================================================
# CHỨC NĂNG 4.1: QUY HOẠCH TUYẾN TÍNH HÌNH HỌC 2D (CORNER-POINT METHOD)
# ==============================================================================

def solve_linear_programming_2d():
    """
    Giải bài toán Quy hoạch Tuyến tính 2D tối đa hóa lượt tiếp cận người dùng:
    Hàm mục tiêu: max f(x1, x2) = 50 * x1 + 40 * x2
    
    Ràng buộc kinh doanh:
    1. Ngân sách tiếp thị:  2 * x1 + 4 * x2 <= 100
    2. Giờ công nhân sự:    3 * x1 + 2 * x2 <= 90
    3. Ràng buộc không âm:  x1 >= 0, x2 >= 0
    
    Phương pháp: Duyệt toàn bộ đỉnh biên (Corner-Point / Vertex Enumeration).
    """
    # 1. Tìm các giao điểm tiềm năng của các đường biên:
    # L1: 2*x1 + 4*x2 = 100
    # L2: 3*x1 + 2*x2 = 90
    # L3: x1 = 0
    # L4: x2 = 0
    candidates = []
    
    # Giao L3, L4: (0, 0)
    candidates.append((0.0, 0.0))
    
    # Giao L1, L4 (x2 = 0 => 2*x1 = 100):
    candidates.append((100.0 / 2.0, 0.0))
    
    # Giao L2, L4 (x2 = 0 => 3*x1 = 90):
    candidates.append((90.0 / 3.0, 0.0))
    
    # Giao L1, L3 (x1 = 0 => 4*x2 = 100):
    candidates.append((0.0, 100.0 / 4.0))
    
    # Giao L2, L3 (x1 = 0 => 2*x2 = 90):
    candidates.append((0.0, 90.0 / 2.0))
    
    # Giao L1, L2:
    # 2*x1 + 4*x2 = 100
    # 3*x1 + 2*x2 = 90 => 6*x1 + 4*x2 = 180
    # Trừ pt: 4*x1 = 80 => x1 = 20 => x2 = (100 - 40) / 4 = 15
    candidates.append((20.0, 15.0))
    
    # 2. Lọc các đỉnh thỏa mãn miền chấp nhận được (Feasible Region)
    feasible_vertices = []
    eps = 1e-6
    for x1, x2 in candidates:
        if x1 >= -eps and x2 >= -eps:
            if (2 * x1 + 4 * x2 <= 100 + eps) and (3 * x1 + 2 * x2 <= 90 + eps):
                pt = (round(x1, 2), round(x2, 2))
                if pt not in feasible_vertices:
                    feasible_vertices.append(pt)
                    
    # 3. Tính giá trị hàm mục tiêu f(x1, x2) tại các đỉnh hợp lệ
    best_value = -float('inf')
    optimal_point = None
    results_table = []
    
    for x1, x2 in feasible_vertices:
        f_val = 50 * x1 + 40 * x2
        results_table.append(((x1, x2), f_val))
        if f_val > best_value:
            best_value = f_val
            optimal_point = (x1, x2)
            
    return optimal_point, best_value, results_table


# ==============================================================================
# CHỨC NĂNG 4.2: ĐƯA VỀ DẠNG CHÍNH TẮC & THIẾT LẬP BÀI TOÁN ĐỐI NGẪU
# ==============================================================================

def get_canonical_and_dual_representation():
    """
    Trả về định nghĩa dạng chính tắc (Standard Form) và bài toán đối ngẫu (Dual Problem).
    """
    canonical_text = """
[1] DẠNG CHÍNH TẮC (STANDARD FORM):
Thêm hai biến bù (Slack Variables) s1 >= 0, s2 >= 0 đại diện cho phần ngân sách và giờ công còn dư:
  Hàm mục tiêu:  max f(x1, x2) = 50*x1 + 40*x2 + 0*s1 + 0*s2
  Các ràng buộc đẳng thức:
    2*x1 + 4*x2 + 1*s1 + 0*s2 = 100  (Đẳng thức ngân sách)
    3*x1 + 2*x2 + 0*s1 + 1*s2 = 90   (Đẳng thức nhân sự)
    x1 >= 0, x2 >= 0, s1 >= 0, s2 >= 0
"""
    dual_text = """
[2] BÀI TOÁN ĐỐI NGẪU (DUAL PROBLEM):
Gọi y1, y2 >= 0 lần lượt là biến đối ngẫu (giá bóng - Shadow Prices) của ràng buộc ngân sách và giờ công:
  Hàm mục tiêu đối ngẫu:  min g(y1, y2) = 100*y1 + 90*y2
  Các ràng buộc đối ngẫu:
    2*y1 + 3*y2 >= 50  (Chi phí biên cho 1 chiến dịch Facebook)
    4*y1 + 2*y2 >= 40  (Chi phí biên cho 1 chiến dịch Google)
    y1 >= 0, y2 >= 0

Theo Định lý Đối ngẫu Mạnh (Strong Duality Theorem):
Giá trị tối ưu của bài toán đối ngẫu min g(y) = max f(x) = 1600.
Tại nghiệm tối ưu: y1 = 2.5, y2 = 15.0 => g(y*) = 100*2.5 + 90*15.0 = 250 + 1350 = 1600.
"""
    return canonical_text, dual_text


# ==============================================================================
# CHỨC NĂNG 4.3: TỐI ƯU HÓA GRADIENT DESCENT 1D
# ==============================================================================

def gradient_descent_1d(w_start=10.0, learning_rate=0.1, max_epochs=50, tolerance=1e-6):
    """
    Tối ưu hóa trọng số w để cực tiểu hóa hàm mất mát chi phí phi tuyến:
    L(w) = w^2 - 6w + 9 = (w - 3)^2
    
    Đạo hàm bậc nhất (Gradient):
    L'(w) = dL/dw = 2w - 6
    
    Quy tắc cập nhật:
    w_{t+1} = w_t - alpha * L'(w_t)
    """
    history = []
    w = w_start
    
    for epoch in range(max_epochs):
        # Tính giá trị hàm mất mát L(w)
        loss = w ** 2 - 6 * w + 9
        # Tính gradient
        grad = 2 * w - 6
        history.append((epoch, round(w, 5), round(loss, 6), round(grad, 5)))
        
        if abs(grad) < tolerance:
            break
            
        # Cập nhật trọng số ngược chiều gradient
        w -= learning_rate * grad
        
    return w, history


# ==============================================================================
# PHẦN MỞ RỘNG DSA: QUY HOẠCH ĐỘNG (KNAPSACK 0/1) & ROLLING HASH
# ==============================================================================

def knapsack_01_promo_combo(weights, values, capacity):
    """
    Quy hoạch động Knapsack 0/1 tối ưu hóa gói combo sản phẩm khuyến mãi:
    - weights: Chi phí ngân sách hỗ trợ cho từng gói sản phẩm
    - values: Lợi nhuận / Điểm đánh giá mang lại
    - capacity: Tổng ngân sách khuyến mãi tối đa
    """
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i = weights[i - 1]
        v_i = values[i - 1]
        for w in range(capacity + 1):
            if w >= w_i:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)
            else:
                dp[i][w] = dp[i - 1][w]

    # Truy vết các sản phẩm được chọn
    selected_items = []
    curr_w = capacity
    for i in range(n, 0, -1):
        if dp[i][curr_w] != dp[i - 1][curr_w]:
            selected_items.append(i - 1)
            curr_w -= weights[i - 1]

    selected_items.reverse()
    return dp[n][capacity], selected_items


def rolling_hash_search(text, pattern, base=256, prime=1000000007):
    """
    Thuật toán Rolling Hash (Rabin-Karp) tìm kiếm nhanh mã giảm giá / chuỗi từ khóa:
    Độ phức tạp trung bình O(N + M).
    """
    n = len(text)
    m = len(pattern)
    if m > n:
        return []

    # Tính hash cho pattern và cửa sổ đầu tiên của text
    pattern_hash = 0
    window_hash = 0
    h = pow(base, m - 1, prime)

    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % prime
        window_hash = (window_hash * base + ord(text[i])) % prime

    match_indices = []
    for i in range(n - m + 1):
        if pattern_hash == window_hash:
            if text[i:i + m] == pattern:
                match_indices.append(i)

        if i < n - m:
            # Dịch chuyển cửa sổ (Rolling Hash)
            window_hash = (window_hash - ord(text[i]) * h) % prime
            window_hash = (window_hash * base + ord(text[i + m])) % prime
            window_hash = (window_hash + prime) % prime

    return match_indices


# ==============================================================================
# HÀM DEMO MODULE 4
# ==============================================================================

def demo_module_4():
    """Hàm chạy demo kiểm thử toàn bộ Chức năng Module 4."""
    print("=" * 70)
    print("DEMO MÔ-ĐUN 4: TỐI ƯU HÓA CHI PHÍ MARKETING & GRADIENT DESCENT")
    print("=" * 70)

    # 1. Quy hoạch tuyến tính hình học 2D
    print("\n--- 4.1. QUY HOẠCH TUYẾN TÍNH HÌNH HỌC 2D (MARKETING ADS) ---")
    opt_pt, max_reach, table = solve_linear_programming_2d()
    print("Các đỉnh biên thuộc miền chấp nhận được (Feasible Vertices) & Giá trị hàm mục tiêu:")
    for pt, val in table:
        marker = " <--- [ĐIỂM TỐI ƯU NHẤT]" if pt == opt_pt else ""
        print(f"  Đỉnh {pt}: Lượt tiếp cận f(x1, x2) = {val}{marker}")
    print(f"\n=> Kết luận nghiệm tối ưu:")
    print(f"  Số chiến dịch Facebook Ads (x1): {opt_pt[0]}")
    print(f"  Số chiến dịch Google Ads (x2):   {opt_pt[1]}")
    print(f"  Lượt tiếp cận tối đa:            {max_reach} nghìn lượt")

    # 2. Dạng chính tắc và Đối ngẫu
    print("\n--- 4.2. DẠNG CHÍNH TẮC & BÀI TOÁN ĐỐI NGẪU ---")
    canonical_repr, dual_repr = get_canonical_and_dual_representation()
    print(canonical_repr)
    print(dual_repr)

    # 3. Gradient Descent 1D
    print("--- 4.3. TỐI ƯU HÓA GRADIENT DESCENT 1D ---")
    print("Hàm mất mát chi phí phi tuyến: L(w) = w^2 - 6w + 9 (Điểm cực tiểu lý thuyết w = 3, L = 0)")
    w_opt, logs = gradient_descent_1d(w_start=10.0, learning_rate=0.2, max_epochs=20)
    print(f"{'Epoch':<8}{'w':<12}{'Loss L(w)':<16}{'Gradient dL/dw':<16}")
    print("-" * 52)
    for ep, w_val, loss_val, grad_val in logs[:10]:
        print(f"{ep:<8}{w_val:<12.5f}{loss_val:<16.6f}{grad_val:<16.5f}")
    if len(logs) > 10:
        last = logs[-1]
        print("...")
        print(f"{last[0]:<8}{last[1]:<12.5f}{last[2]:<16.6f}{last[3]:<16.5f}")
    print(f"\n=> Trọng số hội tụ tối ưu: w* = {round(w_opt, 4)} (Hàm mất mát đạt cực tiểu L(w*) ≈ 0.0)")

    # 4. Mở rộng DSA: Knapsack & Rolling Hash
    print("\n--- PHẦN MỞ RỘNG DSA: COMBO KHUYẾN MÃI (DP) & ROLLING HASH ---")
    weights = [2, 3, 4, 5, 9]     # Chi phí ngân sách từng gói sản phẩm (triệu VNĐ)
    values = [3, 4, 8, 8, 14]     # Doanh số mang lại (triệu VNĐ)
    budget = 10                  # Tổng ngân sách khuyến mãi tối đa
    max_val, items = knapsack_01_promo_combo(weights, values, budget)
    print(f"Knapsack 0/1 - Tối ưu combo khuyến mãi (Ngân sách tối đa {budget}tr):")
    print(f"  Doanh số tối đa đạt được: {max_val} triệu VNĐ")
    print(f"  Các gói combo được chọn: {[f'Gói {idx+1} (chi phí {weights[idx]}tr, doanh số {values[idx]}tr)' for idx in items]}")

    text_logs = "POLY-MART_VOUCHER_SALE20_SUMMER_SALE50_SUPER_SALE20"
    pattern = "SALE20"
    matches = rolling_hash_search(text_logs, pattern)
    print(f"\nRolling Hash - Tra cứu chuỗi mã giảm giá '{pattern}' trong chuỗi nhật ký:")
    print(f"  Vị trí tìm thấy mã: {matches}")


if __name__ == "__main__":
    demo_module_4()

