# DỰ ÁN ASSIGNMENT: TOÁN CHO HỌC MÁY (ITA201)
## HỆ THỐNG SMART DATA PIPELINE SÀN THƯƠNG MẠI ĐIỆN TỬ POLY-MART

---

## 1. Giới thiệu tổng quan
Dự án được xây dựng nhằm giải quyết 4 bài toán kinh doanh và vận hành cốt lõi cho sàn thương mại điện tử **POLY-MART** dựa trên các nền tảng toán học và giải thuật học máy:
1. **Phân khúc khách hàng & Nén hồ sơ mua sắm**: Đại số tuyến tính, chuẩn vector L1/L2, phép chiếu cơ sở và phân tích thành phần chính (PCA thuần).
2. **Lọc đánh giá ảo & Đo lường thông tin thuộc tính**: Sinh tập con đặc trưng (Backtracking), phân loại Naive Bayes với kỹ thuật làm mượt Laplace, và lý thuyết độ hỗn loạn Shannon Entropy / Information Gain.
3. **Mạng lưới người dùng & Dự báo chuỗi mua sắm**: Biểu diễn đồ thị mạng xã hội, duyệt BFS/DFS, kiểm tra đồ thị hai phía, dự báo xác suất chuyển đổi chuỗi Markov và tối ưu hóa mạng giao vận bằng Dijkstra & Kruskal DSU.
4. **Tối ưu hóa ngân sách tiếp thị & Giá bán**: Quy hoạch tuyến tính 2D hình học (Corner-point method), dạng chính tắc & đối ngẫu (Slack variables & Shadow prices), tối ưu hóa hàm mất mát phi tuyến bằng Gradient Descent 1D, kết hợp Quy hoạch động (Knapsack 0/1) và Rolling Hash.

> **Quy chuẩn mã nguồn:** Không sử dụng thư viện giải thuật cao cấp (`numpy`, `networkx`, `scikit-learn`) để làm thay thuật toán cốt lõi. Toàn bộ giải thuật được lập trình thuần bằng thư viện chuẩn của Python (`heapq`, `collections`, `math`, `sys`).

---

## 2. Cấu trúc thư mục dự án

```
Assignment/
├── customer_analytics.py   # Mô-đun 1: Khởi tạo dữ liệu, khoảng cách L1/L2, đổi cơ sở, PCA thuần (Bài 1, 2, 3)
├── review_classifier.py    # Mô-đun 2: Feature selection (Quay lui), Naive Bayes Laplace, Shannon Entropy & IG (Bài 4, 5)
├── behavior_network.py     # Mô-đun 3: Đồ thị BFS/DFS, Markov chuỗi, Bipartite graph, Dijkstra & Kruskal (Bài 6, 7)
├── revenue_optimizer.py    # Mô-đun 4: Quy hoạch tuyến tính 2D, Dạng đối ngẫu, Gradient Descent 1D, Knapsack DP (Bài 8)
├── main_dashboard.py       # Menu Console tương tác trực tiếp
├── main_ecommerce.py       # Điểm kích hoạt chương trình thay thế
└── README.md               # Tài liệu hướng dẫn và báo cáo lý thuyết toán học
```

---

## 3. Hướng dẫn cài đặt và thực thi

### Yêu cầu môi trường
- Python 3.8+ (khuyến nghị Python 3.10 hoặc 3.11).
- Không yêu cầu cài thêm thư viện bên ngoài qua `pip` (sử dụng 100% Python Standard Library).

### Cách chạy hệ thống
Di chuyển vào thư mục dự án:
```powershell
cd "d:\HỌC TẬP\Bài tập\Toán học cho máy\Assignment"
```

1. **Khởi chạy Menu Console tương tác toàn hệ thống:**
   ```powershell
   python main_dashboard.py
   # hoặc
   python main_ecommerce.py
   ```

2. **Chạy kiểm thử độc lập từng mô-đun:**
   - Kiểm thử Mô-đun 1:
     ```powershell
     python customer_analytics.py
     ```
   - Kiểm thử Mô-đun 2:
     ```powershell
     python review_classifier.py
     ```
   - Kiểm thử Mô-đun 3:
     ```powershell
     python behavior_network.py
     ```
   - Kiểm thử Mô-đun 4:
     ```powershell
     python revenue_optimizer.py
     ```

---

## 4. Tóm tắt phân tích toán học các mô-đun

### Mô-đun 1: Tiền xử lý & PCA
- **Chuẩn vector**: Đo lường khoảng cách Manhattan ($L_1 = \sum |u_i - v_i|$) và Euclidean ($L_2 = \sqrt{\sum (u_i - v_i)^2}$) giúp đo khoảng cách tương đồng giữa các khách hàng mà không bị méo lệch tỷ lệ sau khi chuẩn hóa Min-Max về $[0, 1]$.
- **PCA thuần**:
  1. Trừ trung bình cột: $X_{centered} = X - \mu$.
  2. Ma trận hiệp phương sai mẫu: $\text{Cov} = \frac{1}{M-1} X_{centered}^T X_{centered}$.
  3. Tìm vector riêng chính bằng Power Iteration và giảm cấp Hotelling Deflation:
     $$\lambda_1 = v_1^T \text{Cov} v_1, \quad \text{Cov}_2 = \text{Cov} - \lambda_1 v_1 v_1^T$$
  4. Chiếu dữ liệu sang mặt phẳng 2D: $z_{k} = [X_{centered}[k] \cdot PC_1, X_{centered}[k] \cdot PC_2]$.

### Mô-đun 2: Phân loại Naive Bayes & Entropy
- **Naive Bayes với Laplace Smoothing**:
  $$P(w_i | C) = \frac{\text{count}(w_i, C) + 1}{\sum_{w} \text{count}(w, C) + |V|}$$
  Tránh triệt tiêu xác suất về 0 khi gặp từ vựng mới chưa xuất hiện trong tập huấn luyện.
- **Shannon Entropy & Information Gain**:
  $$H(S) = - \sum_{i=1}^k p_i \log_2(p_i)$$
  $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
  Giúp thuật toán tự động chọn thuộc tính giảm độ bất định cao nhất để phân nhánh cây quyết định.

### Mô-đun 3: Mạng xã hội, Chuỗi Markov & Đồ thị hai phía
- **Duyệt đồ thị**: BFS tìm khoảng cách bước ngắn nhất $\mathcal{O}(V + E)$; DFS gom cụm các nhóm người dùng liên thông.
- **Chuỗi Markov**: Ma trận chuyển trạng thái ngẫu nhiên $P$. Trạng thái sau $k$ bước $v_k = v_0 P^k$. Phân phối dừng cân bằng $\pi^* P = \pi^*$ phản ánh tỷ lệ phân bổ dài hạn của người dùng giữa các hành vi Mua hàng, Xem hàng và Rời bỏ.
- **Đồ thị hai phía**: 2-Coloring BFS xác định tính phân hoạch rõ rệt giữa hai tập đỉnh Kho hàng và Cửa hàng.
- **Routing**: Dijkstra dùng min-heap $\mathcal{O}(E \log V)$ tìm đường rẻ nhất; Kruskal dùng DSU $\mathcal{O}(E \log E)$ xây dựng mạng cáp kết nối tối thiểu.

### Mô-đun 4: Tối ưu hóa Marketing & Gradient Descent
- **Quy hoạch tuyến tính 2D**: Duyệt toàn bộ các giao điểm biên, lọc tập đỉnh thuộc miền chấp nhận được $S$, đánh giá $f(x_1, x_2) = 50x_1 + 40x_2$ tại các đỉnh: Nghiệm tối ưu tại $(20, 15)$ đạt mức tiếp cận 1600 nghìn lượt.
- **Dạng chính tắc & Đối ngẫu**:
  - Primal: $\max 50x_1 + 40x_2$ s.t. $2x_1 + 4x_2 + s_1 = 100, 3x_1 + 2x_2 + s_2 = 90$.
  - Dual: $\min 100y_1 + 90y_2$ s.t. $2y_1 + 3y_2 \ge 50, 4y_1 + 2y_2 \ge 40$. Giá bóng tối ưu $y_1 = 2.5, y_2 = 15.0$ cho tổng chi phí biên $\min g(y) = 1600$.
- **Gradient Descent 1D**: $L(w) = w^2 - 6w + 9 \implies \nabla L = 2w - 6$. Cập nhật $w \leftarrow w - \alpha (2w - 6)$ hội tụ chính xác về $w^* = 3.0$ với $L(w^*) = 0$.
- **Knapsack DP**: Bảng quy hoạch động 2D tối ưu hóa lựa chọn gói combo quà tặng đạt doanh số lớn nhất trong định mức ngân sách.

