#tai thu vien bang lenh nay pip install opencv-python scikit-image numpy
import glob
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluate_image_quality(gt_folder, approx_folder):
    # Lấy danh sách tất cả các file ảnh (hỗ trợ png, jpg, bmp)
    valid_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    gt_files = []
    for ext in valid_extensions:
        gt_files.extend(glob.glob(os.path.join(gt_folder, ext)))

    gt_files = sorted(gt_files)

    if not gt_files:
        print(f"[Lỗi] Không tìm thấy ảnh nào trong thư mục: {gt_folder}")
        return

    psnr_list = []
    ssim_list = []

    print("-" * 65)
    print(f"{'Tên ảnh':<25} | {'PSNR (dB)':<15} | {'SSIM':<15}")
    print("-" * 65)

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        approx_path = os.path.join(approx_folder, filename)

        # Kiểm tra xem file tương ứng có tồn tại ở thư mục xấp xỉ không
        if not os.path.exists(approx_path):
            print(
                f"[Cảnh báo] Bỏ qua '{filename}' do không tìm thấy file tương ứng."
            )
            continue

        # Đọc ảnh (đọc dạng Grayscale vì bài test FPGA thường xử lý trên 1 kênh màu)
        # Nếu muốn test ảnh màu RGB, bạn giữ nguyên cv2.IMREAD_COLOR và thiết biến channel_axis=2 trong ssim
        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        img_approx = cv2.imread(approx_path, cv2.IMREAD_GRAYSCALE)

        # Kiểm tra kích thước ảnh có khớp nhau không
        if img_gt.shape != img_approx.shape:
            print(f"[Lỗi] Kích thước ảnh không khớp: {filename}")
            continue

        # Tính PSNR và SSIM
        # data_range=255 vì ảnh 8-bit có dải giá trị [0, 255]
        score_psnr = psnr(img_gt, img_approx, data_range=255)
        score_ssim = ssim(img_gt, img_approx, data_range=255)

        psnr_list.append(score_psnr)
        ssim_list.append(score_ssim)

        print(
            f"{filename:<25} | {score_psnr:<15.4f} | {score_ssim:<15.4f}"
        )

    print("-" * 65)

    # In kết quả trung bình
    if psnr_list and ssim_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(
            f"{'KẾT QUẢ TRUNG BÌNH (' + str(len(psnr_list)) + ' ảnh)':<25} | {avg_psnr:<15.4f} | {avg_ssim:<15.4f}"
        )
        print("-" * 65)
    else:
        print("[Lỗi] Không có cặp ảnh nào hợp lệ để so sánh.")


# ==========================================
# Cấu hình đường dẫn thư mục của bạn ở đây
# ==========================================
if __name__ == "__main__":
    # Thư mục chứa ảnh gốc (chạy bằng bộ nhân chính xác)
    FOLDER_GROUND_TRUTH = "./results_exact"

    # Thư mục chứa ảnh từ bộ nhân xấp xỉ
    FOLDER_APPROXIMATE = "./results_approx"

    evaluate_image_quality(FOLDER_GROUND_TRUTH, FOLDER_APPROXIMATE)