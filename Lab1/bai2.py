"""
LAB 1 - BÀI 2: Tính Chuẩn Vector L1 và L2 (Vector Norms)
Yêu cầu:
- Cho vector sai số e = y - y_hat.
- Viết 2 hàm:
    + norm_l1(v): Chuẩn Manhattan (||v||_1 = sum(|v_i|))
    + norm_l2(v): Chuẩn Euclidean (||v||_2 = sqrt(sum(v_i^2)))
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def norm_l1(v):
    """
    Tính chuẩn Manhattan (L1 Norm) của vector v:
    ||v||_1 = sum(|v_i|)
    """
    total = 0.0
    for x in v:
        total += abs(x)
    return total


def norm_l2(v):
    """
    Tính chuẩn Euclidean (L2 Norm) của vector v:
    ||v||_2 = sqrt(sum(v_i^2))
    """
    sum_sq = 0.0
    for x in v:
        sum_sq += x ** 2
    return sum_sq ** 0.5


if __name__ == "__main__":
    # Test case mẫu
    error_vector = [3, -4]
    
    print("--- BÀI 2: CHUẨN VECTOR L1 VÀ L2 ---")
    print(f"Vector sai số: {error_vector}")
    print(f"L1 Norm (Manhattan): {norm_l1(error_vector)}")  # Kỳ vọng: 7.0
    print(f"L2 Norm (Euclidean): {norm_l2(error_vector)}")  # Kỳ vọng: 5.0
