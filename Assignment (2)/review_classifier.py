"""
MÔ-ĐUN 2: PHÂN LOẠI KHÁCH HÀNG & ĐO LƯỜNG ĐỘ HỖN LOẠN DỮ LIỆU (Bài 4, Bài 5)
Hệ thống Thương mại Điện tử POLY-MART

Nội dung:
- Chức năng 2.1: Sinh Tổ hợp Đặc trưng (Backtracking Feature Selection).
- Chức năng 2.2: Phân loại Naive Bayes phân loại phản hồi Spam / Ham (Laplace Smoothing).
- Chức năng 2.3: Shannon Entropy & Information Gain khi phân nhánh dữ liệu.
"""

import sys
import math
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ==============================================================================
# CHỨC NĂNG 2.1: SINH TỔ HỢP ĐẶC TRƯNG BẰNG QUAY LUI (BACKTRACKING)
# ==============================================================================

def generate_feature_subsets(features):
    """
    Sử dụng thuật toán Quay lui (Backtracking) để sinh toàn bộ các tập con đặc trưng (Power Set).
    
    Tham số:
        features (list): Danh sách các thuộc tính.
        
    Trả về:
        list of list: Toàn bộ 2^N tập con đặc trưng (bao gồm tập rỗng).
    """
    all_subsets = []
    
    def backtrack(start_index, current_path):
        all_subsets.append(list(current_path))
        for i in range(start_index, len(features)):
            current_path.append(features[i])
            backtrack(i + 1, current_path)
            current_path.pop()
            
    backtrack(0, [])
    return all_subsets


# ==============================================================================
# CHỨC NĂNG 2.2: MÔ HÌNH PHÂN LOẠI NAIVE BAYES VỚI LÀM MƯỢT LAPLACE
# ==============================================================================

class NaiveBayesClassifier:
    """
    Mô hình Naive Bayes phân loại văn bản thuần Python (Multinomial Naive Bayes)
    áp dụng kỹ thuật làm mượt Laplace (Add-1 smoothing) để xử lý từ vựng chưa từng xuất hiện.
    """
    def __init__(self):
        self.class_priors = {}
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words_in_class = defaultdict(int)
        self.vocab = set()
        self.classes = []

    def tokenize(self, text):
        """Tách từ đơn giản và chuẩn hóa chữ thường."""
        return text.lower().replace(",", " ").replace(".", " ").replace("!", " ").split()

    def train(self, training_data):
        """
        Huấn luyện mô hình từ danh sách các cặp (text, label).
        training_data = [("tin nhắn...", "spam"), ("đơn hàng...", "ham"), ...]
        """
        total_docs = len(training_data)
        class_doc_counts = defaultdict(int)

        for text, label in training_data:
            class_doc_counts[label] += 1
            words = self.tokenize(text)
            for w in words:
                self.vocab.add(w)
                self.word_counts[label][w] += 1
                self.total_words_in_class[label] += 1

        self.classes = list(class_doc_counts.keys())
        for c in self.classes:
            self.class_priors[c] = class_doc_counts[c] / total_docs

    def predict(self, text):
        """
        Dự đoán nhãn cho văn bản mới dựa trên Max Log-Likelihood:
        argmax_c [ log P(c) + sum_i log P(w_i | c) ]
        với P(w_i | c) = (count(w_i, c) + 1) / (total_words_c + |V|)
        """
        words = self.tokenize(text)
        vocab_size = len(self.vocab)
        best_class = None
        max_log_prob = -float('inf')
        details = {}

        for c in self.classes:
            log_prob = math.log(self.class_priors[c])
            for w in words:
                count_w = self.word_counts[c].get(w, 0)
                # Laplace Smoothing (+1 vào tử, + |V| vào mẫu)
                p_w_given_c = (count_w + 1) / (self.total_words_in_class[c] + vocab_size)
                log_prob += math.log(p_w_given_c)
            details[c] = log_prob
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = c

        return best_class, details


# ==============================================================================
# CHỨC NĂNG 2.3: SHANNON ENTROPY & INFORMATION GAIN
# ==============================================================================

def calculate_entropy(labels):
    """
    Tính độ hỗn loạn Shannon Entropy của một tập nhãn:
    H(S) = - sum(p_i * log2(p_i))
    """
    if not labels:
        return 0.0
        
    total = len(labels)
    counts = defaultdict(int)
    for lbl in labels:
        counts[lbl] += 1
        
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
            
    return round(entropy, 4)


def calculate_information_gain(dataset, feature_index, target_index):
    """
    Tính Độ tăng thông tin (Information Gain - IG) khi phân nhánh dữ liệu theo thuộc tính feature_index:
    IG(S, A) = H(S) - sum( (|S_v| / |S|) * H(S_v) )
    
    Tham số:
        dataset (list of list): Tập dữ liệu mẫu.
        feature_index (int): Chỉ số cột thuộc tính cần phân nhánh.
        target_index (int): Chỉ số cột nhãn mục tiêu.
    """
    total_samples = len(dataset)
    initial_labels = [row[target_index] for row in dataset]
    H_initial = calculate_entropy(initial_labels)
    
    # Gom nhóm dữ liệu theo từng giá trị của thuộc tính phân nhánh
    subsets_by_value = defaultdict(list)
    for row in dataset:
        val = row[feature_index]
        subsets_by_value[val].append(row[target_index])
        
    # Tính kỳ vọng Entropy sau phân nhánh (Conditional Entropy)
    H_conditional = 0.0
    for val, sub_labels in subsets_by_value.items():
        weight = len(sub_labels) / total_samples
        H_conditional += weight * calculate_entropy(sub_labels)
        
    information_gain = H_initial - H_conditional
    return round(information_gain, 4), H_initial, round(H_conditional, 4)


def demo_module_2():
    """Hàm chạy demo kiểm thử toàn bộ Chức năng Module 2."""
    print("=" * 70)
    print("DEMO MÔ-ĐUN 2: PHÂN LOẠI KHÁCH HÀNG & ĐO LƯỜNG ĐỘ HỖN LOẠN DỮ LIỆU")
    print("=" * 70)

    # 1. Sinh tổ hợp đặc trưng (Backtracking)
    features = ['Doanh_so', 'Do_tuoi', 'Tan_suat_mua', 'Danh_gia_sao']
    subsets = generate_feature_subsets(features)
    print(f"\n--- 2.1. SINH TỔ HỢP ĐẶC TRƯNG CHO FEATURE SELECTION ---")
    print(f"Danh sách đặc trưng ban đầu: {features}")
    print(f"Tổng số tập con sinh được (2^{len(features)} = {2**len(features)}): {len(subsets)}")
    print("Ví dụ 6 tập con đầu tiên:")
    for i, s in enumerate(subsets[:6]):
        print(f"  Tập {i+1}: {s}")

    # 2. Phân loại phản hồi Spam / Ham bằng Naive Bayes + Laplace
    print(f"\n--- 2.2. PHÂN LOẠI EMAIL & PHẢN HỒI BẰNG NAIVE BAYES ---")
    training_data = [
        ("nhận quà tặng trúng thưởng miễn phí bấm ngay", "Spam"),
        ("khuyến mãi siêu khủng quà tặng hấp dẫn", "Spam"),
        ("kiếm tiền online nhận thưởng liền tay", "Spam"),
        ("đơn hàng đã được giao thành công cho khách hàng", "Ham"),
        ("sản phẩm chất lượng đóng gói rất cẩn thận", "Ham"),
        ("cảm ơn shop đã hỗ trợ đổi trả đơn hàng", "Ham")
    ]
    nb = NaiveBayesClassifier()
    nb.train(training_data)

    test_reviews = [
        "chúc mừng bạn nhận quà tặng trúng thưởng miễn phí",
        "đơn hàng đóng gói cẩn thận chất lượng tốt"
    ]
    for rev in test_reviews:
        pred_label, details = nb.predict(rev)
        print(f"Đánh giá: '{rev}'")
        print(f"  -> Kết quả dự đoán: [{pred_label}] (Log-likelihood: {details})")

    # 3. Tính Shannon Entropy và Information Gain
    print(f"\n--- 2.3. SHANNON ENTROPY & INFORMATION GAIN (IG) ---")
    # Dataset khảo sát quyết định mua hàng [Độ tuổi, Thu nhập, Nhãn Mua hàng]
    # Cột 0: Độ tuổi (Tre, Trung_nien, Gia)
    # Cột 1: Thu nhập (Cao, Trung_binh, Thap)
    # Cột 2: Quyết định Mua (Co, Khong)
    sample_dataset = [
        ["Tre",        "Cao",       "Khong"],
        ["Tre",        "Cao",       "Khong"],
        ["Trung_nien", "Cao",       "Co"],
        ["Gia",        "Trung_binh","Co"],
        ["Gia",        "Thap",      "Co"],
        ["Gia",        "Thap",      "Khong"],
        ["Trung_nien", "Thap",      "Co"],
        ["Tre",        "Trung_binh","Khong"],
        ["Tre",        "Thap",      "Co"],
        ["Gia",        "Trung_binh","Co"]
    ]
    
    target_labels = [row[2] for row in sample_dataset]
    H_total = calculate_entropy(target_labels)
    print(f"Entropy ban đầu của toàn bộ tập dữ liệu H(S): {H_total} bit")
    
    ig_age, h_init, h_cond_age = calculate_information_gain(sample_dataset, feature_index=0, target_index=2)
    ig_income, _, h_cond_inc = calculate_information_gain(sample_dataset, feature_index=1, target_index=2)
    
    print(f"Information Gain khi phân nhánh theo 'Độ tuổi': {ig_age} bit")
    print(f"Information Gain khi phân nhánh theo 'Thu nhập': {ig_income} bit")
    best_split = "Độ tuổi" if ig_age > ig_income else "Thu nhập"
    print(f"=> Thuộc tính tối ưu nhất để phân nhánh cây quyết định là: '{best_split}' (Độ giảm hỗn loạn cao nhất).")


if __name__ == "__main__":
    demo_module_2()

