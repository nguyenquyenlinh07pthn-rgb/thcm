"""
LAB 4 - BÀI 1: Thuật toán Sinh Chuỗi Nhị phân Độ dài N (Binary Strings Generation)
Yêu cầu:
- Trong bài toán chọn tập đặc trưng (Feature Selection), bit 1 là giữ lại, bit 0 là loại bỏ thuộc tính.
- Viết hàm generate_binary_strings(n) sinh toàn bộ 2^n chuỗi nhị phân theo thứ tự từ điển bằng thuật toán sinh (không dùng đệ quy).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_binary_strings(n):
    """
    Sinh toàn bộ 2^n chuỗi nhị phân theo thứ tự từ điển bằng thuật toán sinh.
    
    Tham số:
        n (int): Độ dài của chuỗi nhị phân.
        
    Trả về:
        list of str: Danh sách 2^n chuỗi nhị phân.
    """
    results = []
    a = [0] * n  # Cấu hình đầu tiên: toàn bit 0 (000...0)
    
    while True:
        # Bước 1: Lưu cấu hình hiện tại
        results.append("".join(str(bit) for bit in a))
        
        # Bước 2: Tìm bit 0 đầu tiên từ phải sang trái
        i = n - 1
        while i >= 0 and a[i] == 1:
            a[i] = 0
            i -= 1
            
        # Bước 3: Điều kiện dừng: không còn bit 0 nào (đã là cấu hình 111...1)
        if i < 0:
            break
            
        # Bước 4: Đổi bit 0 thành 1
        a[i] = 1
        
    return results


if __name__ == "__main__":
    n = 3
    print("--- BÀI 1: THUẬT TOÁN SINH CHUỖI NHỊ PHÂN ---")
    binary_list = generate_binary_strings(n)
    print(f"Tổng số chuỗi nhị phân sinh được (2^{n} = {2**n}): {len(binary_list)}")
    print("Danh sách chuỗi theo thứ tự từ điển:")
    print(binary_list)

