"""
LAB 4 - BÀI 4: Giải Bài toán Xếp N Quân hậu (N-Queens Problem) bằng Quay lui
Đề bài:
- Xếp N quân hậu lên bàn cờ N x N sao cho không có 2 quân hậu nào ăn nhau (không cùng hàng, cùng cột, cùng đường chéo).
- Viết hàm solve_n_queens(n) nhận vào kích thước bàn cờ n, trả về tổng số cách xếp hợp lệ và in ra biểu diễn trực quan.
- Yêu cầu kỹ thuật:
    + Sử dụng thuật toán Quay lui duyệt từng hàng (row từ 0 đến n - 1).
    + Sử dụng 3 tập hợp (set): cols, diag1 (chính: r - c), diag2 (phụ: r + c) để cắt tỉa nhánh cận O(1).
    + Biểu diễn bàn cờ dạng danh sách các chuỗi, ví dụ: ['.Q..', '...Q', 'Q...', '..Q.'].
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def solve_n_queens(n):
    """
    Giải bài toán N-Queens bằng thuật toán Quay lui kết hợp cắt tỉa nhánh cận.
    
    Tham số:
        n (int): Kích thước bàn cờ n x n và số lượng quân hậu.
        
    Trả về:
        tuple: (total_count, all_solutions)
            total_count (int): Tổng số cách xếp hợp lệ.
            all_solutions (list of list of str): Danh sách tất cả các bàn cờ nghiệm.
    """
    solutions = []
    
    # 3 tập hợp giúp kiểm tra an toàn trong O(1)
    cols = set()
    diag1 = set()  # Đường chéo chính: r - c là hằng số
    diag2 = set()  # Đường chéo phụ: r + c là hằng số
    
    # current_board lưu chỉ số cột của quân hậu tại mỗi hàng: current_board[r] = c
    current_board = [-1] * n
    
    def backtrack(r):
        # Điều kiện dừng: Đã xếp thành công n quân hậu từ hàng 0 đến hàng n - 1
        if r == n:
            # Tạo biểu diễn bàn cờ dạng danh sách chuỗi
            board_repr = []
            for row_idx in range(n):
                c_idx = current_board[row_idx]
                row_str = "." * c_idx + "Q" + "." * (n - 1 - c_idx)
                board_repr.append(row_str)
            solutions.append(board_repr)
            return
            
        # Thử đặt quân hậu tại cột c trên hàng r
        for c in range(n):
            d1 = r - c
            d2 = r + c
            
            # Cắt tỉa nhánh cận: Nếu bị trùng cột hoặc trùng đường chéo thì bỏ qua
            if c in cols or d1 in diag1 or d2 in diag2:
                continue
                
            # Đặt quân hậu (Chọn)
            cols.add(c)
            diag1.add(d1)
            diag2.add(d2)
            current_board[r] = c
            
            # Đệ quy sang hàng tiếp theo
            backtrack(r + 1)
            
            # Quay lui (Undo lựa chọn)
            cols.remove(c)
            diag1.remove(d1)
            diag2.remove(d2)
            current_board[r] = -1

    backtrack(0)
    return len(solutions), solutions


if __name__ == "__main__":
    print("--- BÀI 4: BÀI TOÁN XẾP N QUÂN HẬU (N-QUEENS) ---")
    
    # Test với n = 4
    n_test = 4
    count_4, solutions_4 = solve_n_queens(n_test)
    print(f"\n[Bàn cờ {n_test}x{n_test}] Tổng số cách xếp hợp lệ: {count_4}")
    for idx, board in enumerate(solutions_4, 1):
        print(f"\nCách xếp #{idx}:")
        for row in board:
            print("  " + " ".join(row))
            
    # Test với n = 8 (Bài toán 8 quân hậu kinh điển)
    count_8, _ = solve_n_queens(8)
    print(f"\n[Bàn cờ 8x8] Tổng số cách xếp hợp lệ: {count_8} (Kỳ vọng: 92)")

