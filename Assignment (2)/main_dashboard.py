"""
CHƯƠNG TRÌNH CHÍNH ĐIỀU KHIỂN (main_dashboard.py)
Hệ thống Thương mại Điện tử POLY-MART
Môn: Toán cho học máy (ITA201) - FPT Polytechnic

Giao diện điều khiển Console tương tác chạy demo từng mô-đun và chạy toàn bộ pipeline.
"""

import sys
import os

# Đảm bảo đường dẫn import hoạt động chính xác
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from customer_analytics import demo_module_1
from review_classifier import demo_module_2
from behavior_network import demo_module_3
from revenue_optimizer import demo_module_4


def run_full_pipeline():
    """Chạy liên hoàn toàn bộ 4 mô-đun của hệ thống POLY-MART."""
    print("\n" + "#" * 75)
    print("### BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE TỔNG HỢP POLY-MART (E2E DEMO) ###")
    print("#" * 75)

    input("\nNhấn [Enter] để thực thi MÔ-ĐUN 1 (PCA & Khoảng cách)...")
    demo_module_1()

    input("\nNhấn [Enter] để thực thi MÔ-ĐUN 2 (Naive Bayes & Entropy/IG)...")
    demo_module_2()

    input("\nNhấn [Enter] để thực thi MÔ-ĐUN 3 (Đồ thị, Markov & Giao vận)...")
    demo_module_3()

    input("\nNhấn [Enter] để thực thi MÔ-ĐUN 4 (Quy hoạch tuyến tính & Gradient Descent)...")
    demo_module_4()

    print("\n" + "=" * 75)
    print("HOÀN THÀNH XUẤT SẮC TOÀN BỘ PIPELINE SMART DATA TOÁN CHO HỌC MÁY!")
    print("=" * 75)


def display_menu():
    """Hiển thị menu điều khiển theo đúng cấu trúc đề bài quy định."""
    while True:
        print("\n" + "-" * 55)
        print("--- SMART DATA PIPELINE - TOÁN CHO HỌC MÁY (ITA201) ---")
        print("-" * 55)
        print("1. Demo Module 1: Chuẩn hóa khoảng cách & Nén chiều PCA")
        print("2. Demo Module 2: Sinh tập con, Naive Bayes & Tính Entropy/IG")
        print("3. Demo Module 3: Duyệt đồ thị BFS/DFS & Dự báo Markov")
        print("4. Demo Module 4: Phân bổ ngân sách LP & Gradient Descent")
        print("5. Chạy toàn bộ Pipeline tổng hợp")
        print("0. Thoát chương trình")
        print("-" * 55)
        
        choice = input("Lựa chọn của bạn: ").strip()

        if choice == '1':
            demo_module_1()
        elif choice == '2':
            demo_module_2()
        elif choice == '3':
            demo_module_3()
        elif choice == '4':
            demo_module_4()
        elif choice == '5':
            run_full_pipeline()
        elif choice == '0':
            print("\nCảm ơn bạn đã sử dụng hệ thống! Tạm biệt.")
            break
        else:
            print("\n[!] Lựa chọn không hợp lệ, vui lòng nhập số từ 0 đến 5.")


if __name__ == "__main__":
    display_menu()

