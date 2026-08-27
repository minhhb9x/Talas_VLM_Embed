import torch
import time

def occupy_gpu_vram(target_gb):
    """
    Hàm chiếm dụng VRAM của GPU theo đơn vị GB.
    
    Args:
        target_gb (float): Số GB VRAM muốn chiếm dụng. Ví dụ: 2.5, 4.0, 10.0
    """
    if not torch.cuda.is_available():
        print("Không tìm thấy GPU. Vui lòng kiểm tra lại môi trường.")
        return

    device = torch.device("cuda:0")
    print(f"Thiết bị: {torch.cuda.get_device_name(0)}")
    print(f"Đang khởi tạo để chiếm {target_gb} GB VRAM...")

    # 1 GB = 1024 MB = 1024 * 1024 KB = 1024 * 1024 * 1024 Bytes
    bytes_per_gb = 1024 ** 3
    total_elements = int(target_gb * bytes_per_gb)

    try:
        # Khởi tạo tensor với kiểu dữ liệu int8 (1 byte/phần tử)
        # Dùng torch.zeros để ép GPU thực sự ghi dữ liệu vào bộ nhớ ngay lập tức
        dummy_tensor = torch.zeros(total_elements, dtype=torch.int8, device=device)
        
        print(f"✅ Đã chiếm dụng thành công {target_gb} GB VRAM.")
        print("Đang giữ bộ nhớ... Nhấn Ctrl+C để thoát và giải phóng VRAM.")
        
        # Vòng lặp để giữ cho script không bị tắt (giữ nguyên VRAM đã cấp phát)
        while True:
            time.sleep(1)  # Ngủ để không tốn tài nguyên CPU

    except torch.cuda.OutOfMemoryError:
        print(f"❌ Lỗi: GPU của bạn không còn đủ {target_gb} GB VRAM trống để cấp phát.")
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\nĐã nhận lệnh dừng. Đang dọn dẹp bộ nhớ...")
        # Xóa biến và dọn dẹp cache GPU
        del dummy_tensor
        torch.cuda.empty_cache()
        print("✅ Đã trả lại VRAM về trạng thái bình thường.")

# --- CÁCH SỬ DỤNG ---
# Thay đổi số trong ngoặc thành số GB bạn muốn chiếm (chấp nhận số thập phân)
# Ví dụ: chiếm 4.5 GB VRAM
occupy_gpu_vram(20)