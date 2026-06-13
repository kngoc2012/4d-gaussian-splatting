# Cài đặt & Huấn luyện Mô hình 4D Gaussian Splatting trên Google Colab

Đây là kho lưu trữ bài tập Kỹ thuật Lập trình (KTLT) nghiên cứu, cài đặt và thử nghiệm mô hình [4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives](https://arxiv.org/abs/2412.20720) (Paper V2 của nhóm tác giả Fudan-ZVG) nhằm tái tạo và render không gian 3D/4D động từ video thực tế trên môi trường Google Colab.

## 📁 Thành phần mã nguồn

- `4D_Gaussian_Splatting.ipynb`: Notebook mã nguồn chính, triển khai trọn bộ quy trình từ cấu hình môi trường GPU, biên dịch các module kernel CUDA cho đến huấn luyện (training) và tổng hợp xuất video kết quả (rendering).

---

## 🔍 Kiến trúc & Các giai đoạn thực hiện trong Source Code

Toàn bộ quá trình triển khai trong file `4D_Gaussian_Splatting.ipynb` được chia thành 7 giai đoạn nối tiếp nhau theo tiến trình thực hiện bài tập:

### 0️⃣ Kiểm tra GPU & Môi trường CUDA
- Sử dụng lệnh `!nvidia-smi` để xác minh card đồ họa (NVIDIA T4 / V100 / L4 trên Colab).
- Kiểm tra tính tương thích giữa phiên bản PyTorch và CUDA toolkit.

### 1️⃣ Cài đặt Dependencies & Biên dịch Kernel CUDA
- Clone kho dữ liệu mã nguồn gốc [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting).
- Cài đặt công cụ ước lượng pose máy ảnh **COLMAP**.
- Xây dựng và dịch cxx/cuda extensions cho các gói mô-đun cốt lõi:
  - `diff-gaussian-rasterization` (bộ rasterization khả vi được tùy chỉnh cho trường Gaussian 4D)
  - `simple-knn` (thuật toán k-nearest neighbors để khởi tạo bán kính Gaussians)
  - `pointops2` (toán tử thao tác điểm không gian)

### 2️⃣ Nạp Dữ liệu Đầu vào (Video Upload / D-NeRF dataset)
- Hỗ trợ 2 chế độ nạp dữ liệu:
  - **Option A:** Upload trực tiếp video custom của người dùng từ máy cá nhân (quay xoay quanh vật thể tĩnh hoặc chuyển động nhẹ).
  - **Option B:** Sử dụng bộ dataset tổng hợp chuẩn D-NeRF (bouncingballs, lego, hook...).

### 3️⃣ Tiền xử lý Video: Tách Frame & Trích xuất Pose (COLMAP)
- Tự động tách chuỗi khung hình từ video gốc (ví dụ trích xuất 32 frames `.png`).
- Chạy hệ thống **COLMAP** (Feature extraction & Matcher, Mapper) để ước lượng các ma trận góc nhìn máy ảnh (camera intrinsic & extrinsic poses) và điểm đám mây 3D khởi tạo.
- Quy chuẩn định dạng cấu trúc thư mục dữ liệu theo chuẩn đầu vào của 4DGS (`data/custom`).

### 4️⃣ Cấu hình Tham số & Huấn luyện (Training) Mô hình 4DGS
- Thiết lập siêu tham số tối ưu đặc thù cho **Native 4D Primitives** (Bài báo số 2):
  - `gaussian_dim: 4` (Biến thiên không-thời gian trên không gian 4D thuần túy)
  - `rot_4d: True` (Kích hoạt ma trận xoay 4D cho các tập Gaussian bất đẳng hướng)
  - `force_sh_3d: False` (Sử dụng hàm cầu Spherindrical Harmonics 4D thay vì SH 3D tĩnh)
- Khởi chạy quá trình tối ưu hóa (Iterative Training & Splatting optimization), lưu lại các checkpoint tối ưu nhất (`chkpnt_best.pth`).

### 5️⃣ Render Thành quả Video 3D/4D
- Nạp trọng số mô hình từ checkpoint đã huấn luyện thành công.
- Tự động khắc phục lỗi tải trọng số (`_pickle.UnpicklingError: Weights only load failed` do thay đổi bảo mật trên PyTorch 2.6).
- Duyệt qua mảng camera poses (hàm giải nén cấu trúc camera list), render từng khung hình kết quả vào thư mục `renders/` và sử dụng `ffmpeg` ghép nối lại thành video động `.mp4`.

### 6️⃣ Đánh giá Kết quả & Trải nghiệm (Download)
- Hiển thị trực tiếp (inline embedded view) các video so sánh Ground Truth vs. Rendered bên trong Notebook Colab.
- Tạo gói tải về máy trọn bộ thành quả video 3D/4D và mô hình đã được đào tạo.

---

## 🚀 Hướng dẫn Sử dụng & Trải nghiệm

1. Tải file `4D_Gaussian_Splatting.ipynb` về máy hoặc mở trực tiếp trên Google Colab.
2. Chuyển thời gian chạy (Runtime type) sang **T4 GPU** (hoặc cao hơn).
3. Thi hành tuần tự các ô lệnh từ trên xuống dưới (Run all cells).
4. Tại ô tải dữ liệu ở bước 2, nhấp chuột chọn video mẫu của bạn và ngồi thư giãn trong lúc hệ thống tự động biên dịch, huấn luyện và trả về video 3D kết quả tải xuống!

---
*Khoảng thời gian triển khai bài tập: 06/2026*