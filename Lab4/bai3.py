"""
LAB 4 - BÀI 3: Thuật toán Sinh Hoán vị theo Thứ tự Từ điển (Lexicographical Permutations)
Đề bài:
- Viết hàm generate_permutations(n) nhận vào số nguyên dương n, sinh toàn bộ n! hoán vị của tập {1, 2, ..., n}
  theo thứ tự từ điển bằng thuật toán sinh (vòng lặp while thuần, không dùng thư viện itertools).
- Thuật toán sinh hoán vị kế tiếp (Narayana Pandita):
    1. Tìm vị trí i lớn nhất sao cho a[i] < a[i + 1].
    2. Nếu không tìm thấy -> Đạt cấu hình cuối cùng, dừng thuật toán.
    3. Tìm vị trí k lớn nhất sao cho a[k] > a[i].
    4. Đổi chỗ a[i] và a[k].
    5. Lật ngược đoạn từ a[i + 1] đến a[n - 1].
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_permutations(n):
    """
    Sinh toàn bộ n! hoán vị của {1, 2, ..., n} theo thứ tự từ điển bằng thuật toán sinh.
    
    Tham số:
        n (int): Số phần tử cần hoán vị.
        
    Trả về:
        list of list: Danh sách n! hoán vị.
    """
    if n <= 0:
        return []
        
    # Cấu hình khởi đầu: [1, 2, ..., n]
    a = list(range(1, n + 1))
    results = []
    
    while True:
        # Lưu cấu hình hiện tại
        results.append(list(a))
        
        # Bước 1: Tìm vị trí i lớn nhất sao cho a[i] < a[i+1]
        i = n - 2
        while i >= 0 and a[i] >= a[i + 1]:
            i -= 1
            
        # Bước 2: Nếu không tìm thấy (toàn dãy giảm dần) -> Đã là hoán vị cuối cùng
        if i < 0:
            break
            
        # Bước 3: Tìm vị trí k lớn nhất sao cho a[k] > a[i]
        k = n - 1
        while a[k] <= a[i]:
            k -= 1
            
        # Bước 4: Đổi chỗ a[i] và a[k]
        a[i], a[k] = a[k], a[i]
        
        # Bước 5: Lật ngược đoạn từ a[i + 1] đến a[n - 1]
        left = i + 1
        right = n - 1
        while left < right:
            a[left], a[right] = a[right], a[left]
            left += 1
            right -= 1
            
    return results


if __name__ == "__main__":
    n = 3
    print("--- BÀI 3: THUẬT TOÁN SINH HOÁN VỊ TỪ ĐIỂN ---")
    perms = generate_permutations(n)
    
    # Tính giai thừa n!
    fact = 1
    for x in range(1, n + 1):
        fact *= x
        
    print(f"Số hoán vị của tập 1..{n} ({n}! = {fact}): {len(perms)}")
    print("Danh sách hoán vị theo thứ tự từ điển:")
    for idx, p in enumerate(perms, 1):
        print(f"  {idx}: {p}")

