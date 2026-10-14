# Kiến trúc OriWeaveFormer — đặc tả đầy đủ theo mã nguồn

Tài liệu này là bảng kê kiến trúc chuẩn của `src/oriweaveformer/model.py`. Nội dung mô tả **mô hình suy luận**, không gộp augmentation, loss, EMA, sliding-window hay D4 8TTA vào kiến trúc.

## 1. Cấu hình mặc định và đầu vào/đầu ra

```text
encoder             = mit_b3
pretrained          = imagenet
input channels      = 3
encoder depth       = 5
retained features   = (64, 128, 320, 512)
OCFE spans          = (4, 3, 8)
decoder_dim         = 256
decoder dropout     = 0.1
OCFE initial scale  = 0.1
output channels     = 1
```

Đầu vào có dạng `[B,3,H,W]`. Đầu ra là `{"logits": tensor}` có dạng `[B,1,H,W]`; model **không sigmoid logits** ở bên trong.

Với crop 512, bốn đặc trưng giữ lại lần lượt là:

| Ký hiệu | Kích thước | Vai trò |
|---|---:|---|
| `F1` | `[B,64,128,128]` | mức mịn, sinh trường hướng |
| `F2` | `[B,128,64,64]` | mức trung gian |
| `F3` | `[B,320,32,32]` | ngữ nghĩa trung-cao, không đi qua OCFE |
| `F4` | `[B,512,16,16]` | bottleneck sâu nhất |

## 2. Encoder MiT-B3

Encoder được tạo bởi `segmentation_models_pytorch.encoders.get_encoder`, không được định nghĩa lại trong file model. Với MiT-B3, bốn stage dùng embedding dimension `(64,128,320,512)`, số block `(3,4,18,3)`, số attention head `(1,2,5,8)` và spatial-reduction ratio `(8,4,2,1)`.

- Stage 1 dùng overlapping patch embedding `Conv2d(3,64,kernel=7,stride=4,padding=3)` và LayerNorm.
- Stage 2–4 dùng overlapping patch embedding `Conv2d(Cin,Cout,kernel=3,stride=2,padding=1)` và LayerNorm.
- Mỗi transformer block gồm LayerNorm, efficient self-attention có spatial reduction, residual connection, LayerNorm, Mix-FFN có depthwise convolution `3×3`, rồi residual connection.
- Các stage giảm độ phân giải tổng cộng xuống `1/4, 1/8, 1/16, 1/32` ảnh đầu vào.
- Model chỉ lấy bốn feature cuối `self.encoder(image)[-4:]`; output stem trước đó không được đưa vào decoder.

Chi tiết encoder phụ thuộc phiên bản SMP đã khóa trong release; đây là phần mã ngoại vi, không phải lớp tự viết của OriWeaveFormer.

## 3. `OrientHead`

`OrientHead(64,64)` nhận `F1` và gồm đúng các lớp sau:

| Thứ tự | Lớp | Biến đổi kênh |
|---:|---|---:|
| 1 | Conv `3×3`, padding 1, bias | 64 → 64 |
| 2 | BatchNorm2d | 64 |
| 3 | GELU | 64 |
| 4 | Conv `3×3`, padding 1, bias | 64 → 64 |
| 5 | BatchNorm2d | 64 |
| 6 | GELU | 64 |
| 7 | Conv `1×1`, bias | 64 → 3 |

Ba kênh thô được tách thành `p`, `q`, `confidence`. Model thực hiện:

```text
n = sqrt(p² + q² + 1e-6)
p_hat = p / n
q_hat = q / n
c = sigmoid(confidence)
```

`(p_hat,q_hat)` biểu diễn `(cos 2θ, sin 2θ)`; do đó hướng đường không phân biệt chiều tiến/lùi. Model chỉ có **một orientation head**, không có ba head hay head junction riêng.

## 4. `FieldConditionedWeaveAgg`

Module này lấy mẫu feature dọc theo hướng dự đoán. Với số kênh `C`, nhánh fuse là:

```text
concat(feature, context) [2C]
→ Conv1×1(2C, max(32,C/2), bias=True)
→ GELU
→ Conv1×1(max(32,C/2), C, bias=True)
```

Nếu trường hướng khác kích thước feature, `p`, `q`, `c` được nội suy bilinear với `align_corners=False`, sau đó `p,q` được chuẩn hóa lại.

Góc và điểm lấy mẫu:

```text
θ = 0.5 atan2(q_hat, p_hat)
Δx_k = spacing × k / width × cos(θ)
Δy_k = spacing × k / width × sin(θ),  k=1..span
```

Mỗi `k` sinh hai vị trí đối xứng `+Δ` và `−Δ`, nên số mẫu là `2×span`. `grid_sample` dùng bilinear, `padding_mode="zeros"`, `align_corners=True`. Đầu ra aggregation là:

```text
F_agg = F + c × Fuse([F, Context])
```

Lưu ý triển khai quan trọng: grid `[B,S,H,W,2]` được reshape thành `[B,S×H,W,2]`; tensor mẫu sau đó được reshape trực tiếp thành `[B,S,C,H,W]` rồi lấy mean theo `S`. Quy ước trục kế thừa này được giữ nguyên để tương thích checkpoint, dù không tương đương cách hoán vị trục mẫu–kênh thường dùng.

Grid được tạo từ `p.detach()` và `q.detach()`: gradient không truyền từ tọa độ lấy mẫu về orientation head. Tuy nhiên `p,q` vẫn nhận gradient qua scale-gate của decoder; `c` nhận gradient qua cả OCFE và decoder.

## 5. `ResidualOCFE`

Mỗi OCFE bao `FieldConditionedWeaveAgg` và một scalar học được:

```text
α = sigmoid(scale_logit)
output = F + α × (F_agg − F)
```

`scale_logit` được khởi tạo bằng `logit(0.1)`, nên `α=0.1` lúc đầu. Ba vị trí áp dụng:

| Nhánh | Feature | C | span | spacing | số điểm |
|---|---|---:|---:|---:|---:|
| `ocfe_fine` | F1 | 64 | 4 | 4.0 | 8 |
| `ocfe_mid` | F2 | 128 | 3 | 4.0 | 6 |
| `ocfe_deep` | F4 | 512 | 8 | 6.0 | 16 |

`F3` đi thẳng vào decoder, không qua OCFE.

## 6. `ProjectNormAct` và GroupNorm

Mỗi scale được chiếu về 256 kênh bằng:

```text
Conv1×1(Cin,256,bias=False) → GroupNorm(32,256) → GELU
```

Hàm `_group_count` chọn số group lớn nhất không vượt 32 và chia hết số kênh. Với `decoder_dim=256`, cả decoder dùng 32 group, mỗi group 8 kênh. Các feature sau projection được nội suy bilinear về kích thước của `F1` với `align_corners=False`.

## 7. Decoder SegFormer có điều kiện hướng

Đây không phải decoder SegFormer nguyên bản. Nó có hai nhánh fusion song song.

### 7.1 Projection đa tỉ lệ

```text
F1:  64 → 256
F2: 128 → 256
F3: 320 → 256
F4: 512 → 256
```

Sau resize, `joined = concat(F4',F3',F2',F1')` có 1024 kênh (thứ tự coarse-to-fine).

### 7.2 Nhánh chuẩn

```text
joined [1024]
→ Conv1×1(1024,256,bias=False)
→ GroupNorm(32,256)
→ GELU
```

### 7.3 Nhánh scale-gate có điều kiện hướng

Trường `O=concat(p,q,c)` được ghép với `joined`:

```text
[joined,O] [1027]
→ Conv1×1(1027,256,bias=False)
→ GroupNorm(32,256)
→ GELU
→ Conv1×1(256,4,bias=True)
→ softmax theo 4 scale
```

Trọng số `w1..w4` nhân với các projected feature theo thứ tự fine-to-deep:

```text
Adaptive = w1F1' + w2F2' + w3F3' + w4F4'
```

Việc `joined` có thứ tự đảo nhưng phép tổng adaptive dùng thứ tự `F1'..F4'` là đúng theo code hiện hành.

### 7.4 Hợp nhất và phân loại

```text
concat(Standard,Adaptive) [512]
→ Conv3×3(512,256,padding=1,bias=False)
→ GroupNorm(32,256)
→ GELU
→ Dropout2d(0.1)
→ Conv1×1(256,1,bias=True)
→ bilinear resize về H×W, align_corners=False
```

Lớp Conv cuối của `scale_gate` khởi tạo weight chuẩn `N(0,10^-3)` và bias 0. Classifier khởi tạo weight `N(0,10^-2)` và bias 0. Các lớp còn lại dùng khởi tạo mặc định của PyTorch/SMP.

## 8. Luồng forward hoàn chỉnh

```text
RGB image
  → MiT-B3 → F1,F2,F3,F4
  → OrientHead(F1) → p,q,c
  → OCFE(F1,p,q,c), OCFE(F2,p,q,c), F3, OCFE(F4,p,q,c)
  → bốn ProjectNormAct + resize về 1/4
  ├─ concat fusion chuẩn
  └─ orientation-conditioned adaptive scale fusion
  → spatial fusion + classifier
  → resize logits về độ phân giải ảnh
```

## 9. Số tham số

Các số dưới đây ứng với cấu hình mặc định MiT-B3/decoder 256:

| Thành phần | Tham số |
|---|---:|
| MiT-B3 encoder | 44,072,128 |
| OrientHead | 74,307 |
| OCFE fine | 6,241 |
| OCFE mid | 24,769 |
| OCFE deep | 393,985 |
| Bốn projection | 264,192 |
| Scale gate | 264,452 |
| Concat fuse | 262,656 |
| Output fuse | 1,180,160 |
| Classifier | 257 |
| **Tổng** | **46,543,147** |

Scalar `scale_logit` đã được tính trong từng OCFE.

## 10. Thành phần không nằm trong kiến trúc model

- Loss huấn luyện của release là `BCEWithLogits + soft Dice`; loss nằm trong `losses.py`, không nằm trong `model.py`.
- Không có orientation loss, clDice, centerline loss hoặc deep supervision trong model cuối.
- EMA là bản sao trọng số phục vụ huấn luyện/chọn checkpoint, không phải layer inference.
- Random crop, AMP, optimizer, scheduler, DDP và gradient accumulation thuộc pipeline train.
- Sliding-window native resolution, threshold tuning và D4 8TTA thuộc pipeline evaluation.
- Sigmoid và threshold chỉ được dùng khi chuyển logits thành xác suất/mask trong validation hoặc evaluation.

## 11. Danh sách class/hàm đã bao phủ

Tài liệu đã đối chiếu đủ toàn bộ định nghĩa trong `model.py`: `_group_count`, `OrientHead`, `FieldConditionedWeaveAgg`, `ResidualOCFE`, `ProjectNormAct`, `OrientationConditionedSegFormerDecoder` và `OriWeaveFormer`. Phần lặp nội bộ của MiT-B3 được mô tả theo cấu hình của dependency SMP thay vì sao chép mã thư viện vào release.
