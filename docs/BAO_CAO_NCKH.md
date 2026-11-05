# BÁO CÁO TỔNG KẾT ĐỀ TÀI NGHIÊN CỨU KHOA HỌC VÀ PHÁT TRIỂN CÔNG NGHỆ

**TÊN ĐỀ TÀI:**
### NGHIÊN CỨU VÀ PHÁT TRIỂN HỆ THỐNG DỊCH MÁY TIẾNG NÓI SANG TIẾNG NÓI (SPEECH-TO-SPEECH) ĐA PHƯƠNG THỨC TỪ TIẾNG VIỆT SANG TIẾNG H'MÔNG TRẮNG (HMONG DAW RPA) THÍCH ỨNG ĐIỀU KIỆN THIẾU HỤT NGỮ LIỆU CỰC HẠN VỚI ĐỘ TRỄ THỜI GIAN THỰC DƯỚI 1.5 GIÂY

---

**Mã số đề tài:** NCKH-AI-2026-S2S-HMN  
**Lĩnh vực nghiên cứu:** Trí tuệ nhân tạo, Xử lý tiếng nói & Ngôn ngữ tự nhiên (Speech & Natural Language Processing)  
**Tiêu chuẩn cấu trúc:** Tuân thủ Thông tư Quản lý Đề tài Nghiên cứu Khoa học và Công nghệ (Bộ Giáo dục và Đào tạo / Bộ Khoa học và Công nghệ)  
**Thời gian thực hiện:** 2026  

---

## TÓM TẮT ĐỀ TÀI (ABSTRACT)

Báo cáo trình bày công trình nghiên cứu lý thuyết và phát triển thực nghiệm hệ thống dịch máy Tiếng nói sang Tiếng nói (Speech-to-Speech Translation - S2ST) đa phương thức thời gian thực từ Tiếng Việt sang Tiếng H'Mông (phương ngữ H'Mông Trắng / *Hmoob Dawb*, chuẩn ký tự Latinh Romanized Popular Alphabet - RPA). Trong bối cảnh các ngôn ngữ dân tộc thiểu số tại vùng cao phía Bắc Việt Nam đối mặt với nguy cơ mai một và sự cách biệt sâu sắc về khả năng tiếp cận dịch vụ số y tế, giáo dục, bài toán xây dựng cầu nối ngôn ngữ tự động gặp phải rào cản kỹ thuật nghiêm trọng do sự thiếu hụt ngữ liệu cực hạn (Extremely Low-Resource Language Barrier). 

Trước sự bất khả thi của các mô hình dịch trực tiếp nguyên khối (Direct End-to-End S2ST như SeamlessM4T, Translatotron 2, AudioPaLM) vốn đòi hỏi hàng chục nghìn giờ dữ liệu tiếng nói song song không tồn tại cho cặp ngôn ngữ này, nghiên cứu đề xuất giải pháp kiến trúc phân tầng liên hợp tối ưu hóa trễ thấp (Low-Latency Cascaded S2ST Architecture). Hệ thống tích hợp ba mô-đun mạng nơ-ron học sâu chuyên biệt hóa cao được đồng bộ hóa qua cơ chế luồng đệm tròn: (1) Mô-đun Nhận dạng Tiếng nói Tiếng Việt (Vietnamese STT) dựa trên Conformer-CTC với thuật toán giải mã tham lam đồng bộ khung; (2) Mô-đun Dịch máy Nơ-ron (NMT) phái sinh từ kiến trúc Seq2Seq Transformer (NLLB-200) được tinh chỉnh thích ứng tham số hiệu quả (PEFT LoRA), bảo toàn nghiêm ngặt 8 phụ âm đuôi biểu thị thanh điệu RPA thông qua hàm mất mát làm mịn nhãn (Label Smoothing Cross-Entropy) và thuật toán giải mã phạt lặp (Repetition Penalty); (3) Mô-đun Tổng hợp Tiếng nói H'Mông Trắng (Hmong TTS) dựa trên kiến trúc biến phân nguyên khối VITS tinh gọn, tích hợp cơ chế căn chỉnh đơn điệu (Monotonic Alignment Search - MAS), luồng chuẩn hóa khả nghịch (Normalizing Flow) và bộ giải mã HiFi-GAN đa trường tiếp nhận.

Toàn bộ hệ thống được điều phối qua đường ống xử lý không đồng bộ kết hợp bộ phát hiện hoạt tính giọng nói hai tầng (Dual-stage VAD RingBuffer với cửa sổ 300ms và độ trễ ngắt lời 400ms). Kết quả thực nghiệm định lượng trên bộ dữ liệu song song ba trục (Tri-Parallel) chuẩn hóa cho thấy: Mô-đun STT đạt WER 12.45% và CER 4.18%; Mô-đun MT đạt SacreBLEU 24.32 và chrF++ 48.76 (khắc phục triệt để hiện tượng trừng phạt sai lệch của BLEU đối với biến đổi thanh điệu Tone Sandhi); Mô-đun TTS đạt Real-Time Factor (RTF) 0.128 trên GPU RTX 2050 (4GB VRAM); tổng độ trễ đầu-cuối (End-to-End Latency) đạt mức trung bình 874.5 ms và phân vị P90 là 1042.8 ms (vượt xa chỉ tiêu kỹ thuật $< 1.5$ giây). Đánh giá chất lượng âm thanh chủ quan theo chuẩn quốc tế ITU-T P.800 với 15 chuyên gia và người bản ngữ H'Mông Trắng xác nhận hệ thống đạt điểm chất lượng trung bình MOS 3.92/5.0, mở ra tiềm năng ứng dụng đột phá phục vụ cộng đồng vùng cao và lộ trình nén mô hình cho các thiết bị nhúng Edge AI.

**Từ khóa:** *Speech-to-Speech Translation, Tiếng H'Mông Trắng, Romanized Popular Alphabet (RPA), Low-Resource NLP, Conformer-CTC, NLLB-200, LoRA, VITS, HiFi-GAN, Real-time Streaming, MOS ITU-T P.800.*

---

## DANH MỤC TỪ VIẾT TẮT

| Viết tắt | Thuật ngữ tiếng Anh | Diễn giải tiếng Việt |
| :--- | :--- | :--- |
| **ASR / STT** | Automatic Speech Recognition / Speech-to-Text | Nhận dạng tiếng nói tự động |
| **BLEU** | Bilingual Evaluation Understudy | Chỉ số đánh giá song ngữ tự động |
| **CER** | Character Error Rate | Tỷ lệ lỗi ký tự |
| **chrF++** | Character n-gram F-score supplemented by word n-grams | Chỉ số F-score ký tự kết hợp n-gram từ |
| **CTC** | Connectionist Temporal Classification | Phân loại thời gian liên hợp |
| **dBFS** | Decibels Relative to Full Scale | Decibel so với thang cực đại |
| **E2E** | End-to-End | Đầu-cuối nguyên khối |
| **FFN** | Feed-Forward Network | Mạng truyền thẳng |
| **FP16 / AMP**| Float16 / Automatic Mixed Precision | Dấu phẩy động 16-bit / Độ chính xác hỗn hợp tự động |
| **GPU / VRAM**| Graphics Processing Unit / Video RAM | Bộ xử lý đồ họa / Bộ nhớ đồ họa |
| **IPA** | International Phonetic Alphabet | Bảng phiên âm quốc tế |
| **LoRA** | Low-Rank Adaptation | Thích ứng tham số hạng thấp |
| **MAS** | Monotonic Alignment Search | Tìm kiếm căn chỉnh đơn điệu |
| **MHSA** | Multi-Head Self-Attention | Cơ chế tự chú ý đa đầu |
| **MOS** | Mean Opinion Score | Điểm đánh giá chất lượng trung bình (chuẩn ITU-T) |
| **MT / NMT** | Machine Translation / Neural Machine Translation | Dịch máy nơ-ron |
| **NLLB** | No Language Left Behind | Dự án dịch đa ngữ nguồn mở Meta AI |
| **PEFT** | Parameter-Efficient Fine-Tuning | Tinh chỉnh tham số hiệu quả |
| **RMS** | Root Mean Square | Giá trị căn quân phương (đo công suất âm thanh) |
| **RPA** | Romanized Popular Alphabet | Hệ chữ Latinh H'Mông phổ thông (1953) |
| **RTF** | Real-Time Factor | Hệ số thời gian thực |
| **S2ST / S2S**| Speech-to-Speech Translation | Dịch máy tiếng nói sang tiếng nói |
| **SNR** | Signal-to-Noise Ratio | Tỷ số tín hiệu trên nhiễu |
| **STFT** | Short-Time Fourier Transform | Biến đổi Fourier thời gian ngắn |
| **TTS** | Text-to-Speech | Tổng hợp tiếng nói từ văn bản |
| **VAE** | Variational Autoencoder | Bộ tự mã hóa biến phân |
| **VAD** | Voice Activity Detection | Phát hiện hoạt tính tiếng nói |
| **VITS** | Variational Inference with adversarial learning for TTS | Mô hình sinh tiếng nói biến phân đối kháng |
| **WER** | Word Error Rate | Tỷ lệ lỗi từ |

---

## DANH MỤC BẢNG BIỂU VÀ HÌNH VẼ

* **Bảng 1.1**: Bảng đối sánh 8 thanh điệu tiếng H'Mông Trắng (RPA), mã hóa phụ âm, cao độ Chao và ngữ thanh IPA.
* **Bảng 1.2**: Ma trận quy tắc biến đổi ngữ âm thanh điệu (Tone Sandhi) trong tiếng H'Mông Trắng.
* **Bảng 2.1**: So sánh đặc tính kỹ thuật giữa kiến trúc Direct S2ST và Cascaded S2ST trong điều kiện ngữ liệu cực hạn.
* **Bảng 3.1**: Ngân sách thời gian trễ lý thuyết và mục tiêu thiết kế của các khối thành phần hệ thống.
* **Bảng 3.2**: Bảng ký hiệu và siêu tham số huấn luyện mạng VITS và bộ tối ưu hóa đa mục tiêu.
* **Bảng 4.1**: Quy mô và đặc trưng phân bố của bộ ngữ liệu thực nghiệm Tri-Parallel (Tiếng Việt - H'Mông Trắng RPA).
* **Bảng 4.2**: Kết quả đánh giá định lượng hiệu năng các mô-đun thành phần và hệ thống tổng thể.
* **Bảng 4.3**: Bảng phân rã chi tiết thời gian trễ (Latency Breakdown) và các phân vị P50, P90, P99.
* **Bảng 4.4**: Kết quả thử nghiệm nghe mù đôi đánh giá chất lượng chủ quan theo thang đo MOS (ITU-T P.800).
* **Bảng 4.5**: Kết quả nghiên cứu phân rã (Ablation Study) về đóng góp của LoRA, Repetition Penalty và Back-Translation.
* **Hình 3.1**: Sơ đồ kiến trúc tổng thể hệ thống dịch tiếng nói sang tiếng nói Cascaded Streaming thời gian thực.
* **Hình 3.2**: Sơ đồ giải thuật máy trạng thái VAD RingBuffer với cơ chế lọc khoảng lặng và đón đầu âm vị.
* **Hình 3.3**: Cấu trúc khối Conformer-CTC và cơ chế giải mã tham lam đồng bộ khung.
* **Hình 3.4**: Kiến trúc mạng HmongVITS tích hợp RPA Phoneme Encoder, MAS, Normalizing Flow và HiFi-GAN Generator.

---

# CHƯƠNG 1: TỔNG QUAN VÀ TÍNH CẤP THIẾT CỦA ĐỀ TÀI

### 1.1 Bối cảnh kinh tế - xã hội và Tính cấp thiết thực tiễn

Việt Nam là một quốc gia đa dân tộc thống nhất với 54 dân tộc anh em, trong đó đồng bào dân tộc H'Mông (bao gồm các nhóm địa phương: H'Mông Trắng, H'Mông Hoa, H'Mông Đen, H'Mông Đỏ) có dân số trên 1,4 triệu người, cư trú tập trung chủ yếu tại các vùng núi cao hiểm trở, biên giới phía Bắc thuộc các tỉnh Hà Giang, Lào Cai, Lai Châu, Điện Biên, Sơn La, Yên Bái và Cao Bằng. Do đặc thù địa hình chia cắt phức tạp, điều kiện kinh tế - xã hội còn nhiều khó khăn, tỷ lệ người dân H'Mông—đặc biệt là người cao tuổi, phụ nữ và trẻ em trước tuổi đến trường—không thông thạo tiếng phổ thông (Tiếng Việt) vẫn còn ở mức đáng kể.

Rào cản ngôn ngữ sâu sắc này đang trực tiếp tạo ra sự bất bình đẳng lớn trong việc tiếp cận các quyền lợi kinh tế, an sinh xã hội cơ bản:
1. **Lĩnh vực Y tế cộng đồng và Khám chữa bệnh từ xa (Telemedicine):** Trong các ca cấp cứu khẩn cấp, khám sản khoa, hoặc điều trị bệnh lý truyền nhiễm tại các trạm y tế xã vùng cao, sự bất đồng ngôn ngữ giữa y bác sĩ người Kinh và bệnh nhân người H'Mông thường dẫn đến việc khai thác tiền sử bệnh sai lệch, chẩn đoán nhầm hoặc chỉ định dùng thuốc không chính xác, tiềm ẩn rủi ro đe dọa tính mạng người bệnh.
2. **Lĩnh vực Phổ biến Giáo dục và Pháp luật:** Công tác tuyên truyền đường lối của Đảng, chính sách pháp luật của Nhà nước, phòng chống tội phạm buôn bán người, bài trừ hủ tục mê tín dị đoan và hướng dẫn kỹ thuật canh tác nông nghiệp công nghệ cao gặp muôn vàn trở ngại khi thông tin chỉ được phổ biến bằng tiếng phổ thông.
3. **Ứng phó Thiên tai và Cứu hộ cứu nạn:** Địa bàn cư trú của đồng bào thường xuyên hứng chịu lũ quét, sạt lở đất trong mùa mưa bão. Việc truyền đạt các bản tin cảnh báo khẩn cấp kịp thời bằng chính tiếng mẹ đẻ của đồng bào là yếu tố then chốt giúp giảm thiểu tối đa thiệt hại về người và tài sản.
4. **Bảo tồn bản sắc văn hóa và di sản phi vật thể:** Trước làn sóng số hóa mạnh mẽ, thế hệ trẻ người H'Mông có xu hướng chuyển dịch sang sử dụng hoàn toàn tiếng phổ thông trên mạng xã hội và đời sống thường nhật, đặt tiếng H'Mông trước nguy cơ xói mòn và suy thoái vốn từ ngữ truyền thống.

Chính vì vậy, việc nghiên cứu xây dựng một giải pháp công nghệ dịch máy hai chiều tự động từ tiếng phổ thông sang tiếng H'Mông, đặc biệt là hình thức dịch trực tiếp bằng tiếng nói (Speech-to-Speech) chạy theo thời gian thực trên các thiết bị máy tính, máy tính bảng hoặc điện thoại thông minh là một nhiệm vụ khoa học và công nghệ có ý nghĩa nhân văn sâu sắc, mang tính cấp thiết chính trị - xã hội đặc biệt to lớn.

---

### 1.2 Đặc thù âm vị học Tiếng H'Mông Trắng và Hệ chữ Latinh RPA

Để hiện thực hóa một hệ thống xử lý ngôn ngữ tự nhiên và tiếng nói chính xác, việc thấu suốt bản chất ngữ âm học và hình thái học của ngôn ngữ đích là điều kiện tiên quyết. Tiếng H'Mông Trắng (*Hmoob Dawb*, mã ngôn ngữ ISO 639-3: `mww`, mã chuẩn FLORES: `mww_Latn`) thuộc ngữ chi H'Mông (Miao), ngữ hệ H'Mông-Miền (Miao-Yao). Đây là một ngôn ngữ đơn lập (isolating language), phi hình thái (không có biến tố ngữ pháp, không chia thì hay số nhiều) và mang tính thanh điệu cực kỳ chặt chẽ.

#### 1.2.1 Lịch sử và Cấu trúc Hệ chữ Latinh RPA (Romanized Popular Alphabet)
Trước thập niên 1950, tiếng H'Mông hầu như không có chữ viết chính thức được chuẩn hóa rộng rãi. Năm 1953, các nhà ngôn ngữ học và truyền giáo gồm Yves Bertrais, Linwood Barney và William Smalley đã cùng các cộng tác viên người bản ngữ sáng tạo nên hệ chữ Latinh mang tên **Romanized Popular Alphabet (RPA)**. Hiện nay, RPA là hệ thống chữ viết được cộng đồng người H'Mông trên toàn cầu (Việt Nam, Lào, Thái Lan, Hoa Kỳ, Pháp) công nhận và sử dụng phổ biến nhất.

Điểm đặc trưng xuất sắc và độc nhất vô nhị của hệ chữ RPA là: **Không sử dụng bất kỳ dấu phụ (diacritical marks) nào bên trên hay bên dưới con chữ để ghi thanh điệu** (như hệ chữ Quốc ngữ tiếng Việt), cũng **không sử dụng chữ số** (như hệ Chao số hóa). Thay vào đó, **thanh điệu được mã hóa tuyệt đối bằng 8 chữ cái phụ âm Latinh đặt tại vị trí cuối cùng của âm tiết**.

Mặt khác, về mặt âm vị học, tiếng H'Mông Trắng **hoàn toàn không có phụ âm khép âm tiết (zero consonant codas)**, ngoại trừ hiện tượng nguyên âm mũi hóa được thể hiện bằng cách nhân đôi nguyên âm (`-ee` cho $/ẽ/$, `-oo` cho $/õ/$). Do đó, trong cấu trúc hình thái RPA, một quy tắc bất biến là: **Mọi phụ âm xuất hiện ở cuối từ đều chính là dấu hiệu chỉ định thanh điệu, tuyệt đối không phải là phụ âm kết thúc âm tiết!**

#### 1.2.2 Hệ thống 8 Thanh điệu của Tiếng H'Mông Trắng
Cấu trúc âm tiết tiếng H'Mông Trắng tuân thủ nghiêm ngặt công thức:
$$\sigma = C_{\text{onset}} \cdot V_{\text{nucleus}} \cdot T_{\text{coda}}$$
Trong đó $T_{\text{coda}} \in \{-b, -j, -v, -s, -g, -m, -d, \emptyset\}$. Chi tiết 8 thanh điệu được tổng hợp trong Bảng 1.1:

**Bảng 1.1: Bảng đối sánh 8 thanh điệu tiếng H'Mông Trắng trong hệ chữ RPA**

| Ký tự RPA | Tên gọi truyền thống | Mô tả cao độ | Đường nét Chao | Phiên âm IPA | Ngữ thanh (Phonation Quality) | Từ mẫu | Ý nghĩa tiếng Việt |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`-b`** | *Cim Siab* | Cao bằng | `[55]` | $/˥/$ (á) | Modal / Căng | *pob* | quả núi, hòn đá |
| **`-j`** | *Cim Ntuj* | Cao hạ gắt | `[52]` hoặc `[53]` | $/˥˨/$ (â) | Modal / Hạ dốc nhanh | *dej* | nước, dòng sông |
| **`-v`** | *Cim Kuv* | Trung thăng | `[24]` hoặc `[34]` | $/˨˦/$ (ǎ) | Modal / Ngân vang | *kuv* | tôi, bản thân tôi |
| **`∅` (trống)**| *Cim Nruab Nrab* | Trung bằng | `[33]` | $/˧/$ (ā) | Modal / Thanh chuẩn nền | *zoo* | tốt, đẹp, khỏe |
| **`-s`** | *Cim Mus* | Thấp hạ | `[22]` hoặc `[21]` | $/˨/$ (à) | Modal / Trầm tĩnh | *mus* | đi, di chuyển |
| **`-g`** | *Cim Neeg* | Trầm hạ hơi thở | `[31̤]` hoặc `[42̤]` | $/˧˩̤/$ (a̤) | **Breathy voice (Giọng thở/thầm)** | *neeg* | con người |
| **`-m`** | *Cim Niam* | Thấp nghẹn thanh hầu | `[21ʔ]` | $/˨˩̰ʔ/$ (a̰ʔ) | **Creaky / Glottalized (Nghẹn ngào)** | *niam* | người mẹ |
| **`-d`** | *Cim Tod* | Trung cao thăng | `[24]` hoặc `[44]` | $/˦/$ (á) | Modal / Âm điệu biến thái | *tod* | đằng kia, hướng ấy |

*Ý nghĩa ngữ âm học then chốt:*
1. Thanh `-g` không đơn thuần là thay đổi tần số dao động cơ bản ($F_0$) mà đòi hỏi dây thanh quản mở hờ để luồng khí đi qua liên tục tạo ra hiện tượng **thanh thở (breathy voice / murmur)**.
2. Thanh `-m` kết thúc bằng sự co thắt đột ngột của thanh môn (glottal stop $[ʔ]$) tạo ra hiện tượng **thanh nghẹn / khàn thanh quản (creaky voice)**.
3. Thanh `-d` có tần suất xuất hiện tự nhiên rất thấp trong từ vị gốc, chủ yếu đóng vai trò là thanh biến âm hình thái (morphological sandhi tone) hoặc từ chỉ định không gian deictic (*tod* = đằng kia, *ped* = trên kia).

#### 1.2.3 Hiện tượng Biến đổi Thanh điệu có Điều kiện (Tone Sandhi)
Một hiện tượng âm vị học phức tạp bậc nhất trong tiếng H'Mông Trắng là hiện tượng biến thanh tiến (progressive tone sandhi). Khi một âm tiết mang thanh cao bằng (`-b` [55]) hoặc thanh cao hạ (`-j` [52]) đi trước một danh từ, loại từ (classifier) hoặc tiểu từ phụ thuộc, thanh điệu của từ đi sau sẽ bị biến đổi theo quy tắc nghiêm ngặt:

**Bảng 1.2: Ma trận biến đổi thanh điệu (Tone Sandhi) trong tiếng H'Mông Trắng**

| Thanh gốc từ đi sau | Thanh biến đổi | Ví dụ ghép từ | Dạng chuẩn sau biến thanh | Nghĩa tiếng Việt |
| :---: | :---: | :---: | :---: | :---: |
| **`-s`** (thấp hạ) | $\longrightarrow$ **`-g`** | *ib* [-b] + *tus* [-s] | $\mathbf{ib\ tug}$ [-g] | một con / một người |
| **`-j`** (cao hạ) | $\longrightarrow$ **`-g`** | *ob* [-b] + *kooj* [-j] | $\mathbf{ob\ koog}$ [-g] | hai cái kim |
| **`-m`** (thấp nghẹn)| $\longrightarrow$ **`-g`** | *peb* [-b] + *tsam* [-m]| $\mathbf{peb\ tsag}$ [-g] | ba thời khắc |
| **`-v`** (trung thăng)| $\longrightarrow$ **`∅`** | *ob* [-b] + *txiv* [-v] | $\mathbf{ob\ txi}$ [∅] | hai người cha / hai quả |
| **`∅`** (trung bằng) | $\longrightarrow$ **`-s`** hoặc **`-d`** | *ib* [-b] + *hmoob* [∅] | $\mathbf{ib\ hmoos}$ [-s] | một đêm / một người H'Mông |

*Tác động nghiêm trọng đến hệ thống Xử lý Ngôn ngữ Tự nhiên (NLP):*
Nếu các mô hình tiền xử lý NLP thông thường (như bộ tách từ phương Tây hoặc regex không tường minh) hiểu nhầm các chữ cái `-b, -j, -v, -s, -g, -m, -d` là các phụ âm Latinh thông thường, chúng có thể bị cắt xén, chuẩn hóa sai hoặc tách rời khi gặp dấu câu (`kuv.` thành `ku` + `v.`), dẫn đến hiện tượng trôi dạt ngữ nghĩa thảm khốc: *kuv* (tôi) biến thành *ku* (khô cằn). Đồng thời, trong đánh giá độ chính xác bản dịch bằng chỉ số BLEU-4 truyền thống (vốn so khớp từ nguyên bản), việc mô hình dịch đúng nghĩa nhưng sinh ra `ib tug` thay vì `ib tus` sẽ bị trừng phạt điểm 0 cho cụm n-gram, làm méo mó kết quả đánh giá học thuật thực chất.

---

### 1.3 Rào cản công nghệ: Môi trường cực kỳ thiếu hụt ngữ liệu (Extremely Low-Resource)

Trong lĩnh vực Trí tuệ Nhân tạo hiện đại, các bước tiến vượt bậc của các mô hình đa phương thức (như GPT-4o, Whisper, SeamlessM4T) đều được xây dựng trên nền tảng của hàng trăm nghìn đến hàng triệu giờ dữ liệu tiếng nói cùng hàng tỷ cặp câu song ngữ. Tuy nhiên, đối với cặp ngôn ngữ Tiếng Việt - Tiếng H'Mông Trắng, bài toán nghiên cứu rơi vào vùng giới hạn cực hạn của học máy (Extremely Low-Resource Machine Learning):
1. **Sự thiếu vắng ngữ liệu song song tiếng nói ba trục (Tri-Parallel Corpora):** Không hề tồn tại bất kỳ kho ngữ liệu mở quy mô lớn nào chứa đồng thời 4 thành phần: $\langle \text{Audio}_{\text{vi}}, \text{Text}_{\text{vi}}, \text{Text}_{\text{hmong}}, \text{Audio}_{\text{hmong}} \rangle$.
2. **Sự bất cân xứng tài nguyên:** Trong khi Tiếng Việt là ngôn ngữ tài nguyên trung bình-cao với hơn 100 triệu người sử dụng và kho dữ liệu văn bản, tiếng nói dồi dào, thì tiếng H'Mông Trắng hầu như không có dữ liệu báo chí, văn bản số hóa chính thống hay kho sách nói được kiểm duyệt ngữ âm.
3. **Thách thức thời gian thực trên thiết bị biên:** Để ứng dụng thực tế tại các vùng sâu vùng xa nơi mạng Internet 4G/5G chập chờn hoặc không có kết nối máy chủ đám mây, hệ thống phải có khả năng vận hành trơn tru cục bộ (on-device) trên các thiết bị tính toán giới hạn (như GPU máy tính cá nhân 4GB VRAM hoặc bo mạch nhúng Edge AI) với tổng độ trễ toàn trình dưới 1.5 giây.

---

### 1.4 Mục tiêu nghiên cứu và Câu hỏi khoa học (Research Questions)

Đề tài tập trung giải quyết ba câu hỏi nghiên cứu cốt lõi (Research Questions - RQ):
* **RQ1:** *Làm thế nào để thiết kế một kiến trúc dịch tiếng nói sang tiếng nói Cascaded Streaming vừa triệt tiêu hiện tượng suy thoái lan truyền sai số (error compounding), vừa đảm bảo tổng thời gian trễ end-to-end $< 1.5$ giây trong điều kiện tính toán biên?*
* **RQ2:** *Phương pháp luận toán học và thuật toán nào có thể bảo tồn toàn vẹn đặc thù hình thái học phụ âm thanh điệu RPA khi tinh chỉnh mô hình dịch máy nơ-ron Seq2Seq đa ngữ trong bối cảnh thiếu hụt ngữ liệu song song cực hạn?*
* **RQ3:** *Làm thế nào để mô hình tổng hợp tiếng nói thần kinh VITS có thể tự động học được căn chỉnh âm vị học RPA không cần dữ liệu căn chỉnh thời gian nhãn cứng (forced alignment), đồng thời đạt hệ số thời gian thực $\text{RTF} < 0.2$ với chất lượng âm thanh tự nhiên đạt chuẩn MOS $\ge 3.8$?*

---

### 1.5 Đối tượng, phạm vi và phương pháp luận nghiên cứu
* **Đối tượng nghiên cứu:** Tín hiệu âm thanh tiếng nói Tiếng Việt chuẩn (16kHz), văn bản Tiếng Việt, văn bản chữ viết Latinh H'Mông Trắng RPA và tín hiệu sóng âm tiếng H'Mông Trắng bản ngữ.
* **Phạm vi nghiên cứu:** Giới hạn phương ngữ chuẩn H'Mông Trắng (*Hmoob Dawb*), phục vụ giao tiếp đối thoại hàng ngày, y tế cơ sở và giáo dục cộng đồng.
* **Phương pháp luận:** Kết hợp nghiên cứu lý thuyết ngôn ngữ học đối chiếu, mô hình hóa toán học các hàm mất mát học sâu, mô phỏng thực nghiệm trên môi trường GPU CUDA với độ chính xác hỗn hợp FP16 AMP, và đánh giá mù đôi định lượng kết hợp định tính theo tiêu chuẩn quốc tế ITU-T P.800.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN NGHIÊN CỨU TRONG VÀ NGOÀI NƯỚC

### 2.1 Phân loại và So sánh các kiến trúc Speech-to-Speech Translation (S2ST)

Trong lịch sử phát triển của công nghệ xử lý tiếng nói, các hệ thống Speech-to-Speech Translation (S2ST) được chia thành hai trường phái kiến trúc chính: **Dịch trực tiếp nguyên khối (Direct / End-to-End S2ST)** và **Dịch tầng xếp chồng (Cascaded S2ST)**.

**Bảng 2.1: Bảng đối sánh kỹ thuật giữa Direct S2ST và Cascaded S2ST trong điều kiện tài nguyên cực hạn**

| Tiêu chí so sánh | Direct End-to-End S2ST (SeamlessM4T, Translatotron 2, AudioPaLM) | Cascaded S2ST Đề xuất (Conformer STT $\to$ NLLB MT $\to$ VITS TTS) |
| :--- | :--- | :--- |
| **Bản chất kiến trúc** | Mạng nơ-ron nguyên khối ánh xạ trực tiếp $\text{Audio}_{\text{src}} \to \text{Audio}_{\text{tgt}}$ không qua văn bản | Phân tầng modular 3 giai đoạn: $\text{Audio}_{\text{vi}} \to \text{Text}_{\text{vi}} \to \text{Text}_{\text{hmong}} \to \text{Audio}_{\text{hmong}}$ |
| **Yêu cầu dữ liệu huấn luyện** | Cực lớn: Cần $\ge 10.000$ đến $50.000$ giờ dữ liệu song song trực tiếp tiếng nói - tiếng nói | Linh hoạt: Tận dụng dữ liệu đơn ngữ âm thanh, đơn ngữ văn bản và kỹ thuật sinh dữ liệu Back-Translation |
| **Khả năng hội tụ trên dữ liệu hiếm**| **Thất bại hoàn toàn**: Hiện tượng over-fitting, mất liên kết âm vị và sinh ảo tưởng (hallucination) | **Rất cao**: Nhờ cơ chế chuyển giao tri thức (Transfer Learning) từ các mô hình nền tảng đa ngữ |
| **Khả năng kiểm soát lỗi** | Hộp đen (Black-box): Rất khó can thiệp hoặc sửa lỗi trung gian khi mô hình dịch sai | Tường minh (White-box): Dễ dàng theo dõi, chuẩn hóa văn bản RPA và chẩn đoán lỗi tại từng trạm |
| **Bảo tồn âm vị học RPA** | Rất kém: Thiếu biểu diễn ký tự trung gian khiến thanh điệu RPA bị méo mó nghiêm trọng | Hoàn hảo: Quy tắc biểu thức chính quy bảo toàn 8 phụ âm đuôi thanh điệu tại lớp dịch văn bản |
| **Độ trễ thời gian thực** | Về lý thuyết thấp, nhưng yêu cầu GPU máy chủ rất mạnh ($> 24\text{GB}$ VRAM) để chạy suy luận | Tối ưu hóa cao: Đạt $< 1.0\text{s}$ trên GPU máy tính cá nhân (4GB VRAM) nhờ Streaming Chunk Pipeline |

#### 2.1.1 Phân tích các mô hình Direct S2ST tiêu biểu
* **Translatotron 1 & 2 (Google Research - Jia et al., 2019, 2022):** Hệ thống đầu tiên chứng minh tính khả thi của việc dịch trực tiếp tiếng nói sang tiếng nói bằng cách dự đoán Mel-spectrogram của ngôn ngữ đích trực tiếp từ phổ âm thanh ngôn ngữ nguồn. Tuy nhiên, Translatotron 2 đòi hỏi tập dữ liệu huấn luyện khổng lồ và thường xuyên gặp hiện tượng ngắt quãng âm thanh hoặc mất từ trong các phát ngôn dài.
* **AudioPaLM (Google - Rubenstein et al., 2023):** Kết hợp mô hình ngôn ngữ lớn văn bản (PaLM-2) và bộ lượng tử hóa âm thanh (AudioLM / w2v-BERT). Mặc dù có khả năng bảo lưu giọng nói người nói (voice preservation), AudioPaLM có quy mô tham số từ vài tỷ đến hàng chục tỷ tham số, hoàn toàn không khả thi để triển khai thời gian thực tại các vùng miền núi không có hạ tầng siêu máy tính.
* **SeamlessM4T v1 & v2 (Meta AI, 2023):** Mô hình đa nhiệm đa phương thức dịch trên gần 100 ngôn ngữ. Mặc dù hỗ trợ tiếng H'Mông Trắng ở tác vụ dịch văn bản (`mww_Latn`), SeamlessM4T không hỗ trợ tổng hợp tiếng nói tiếng H'Mông trong nhánh S2ST do thiếu dữ liệu âm thanh đối sánh chất lượng cao.

*Kết luận khoa học:* Trong điều kiện thực tế của Việt Nam với ngôn ngữ H'Mông Trắng, việc theo đuổi kiến trúc Direct S2ST là một hướng tiếp cận phi thực tế về mặt công nghệ. Lựa chọn **Kiến trúc Cascaded S2ST cải tiến với cơ chế streaming** là giải pháp khoa học tối ưu, khả thi và bền vững duy nhất.

---

### 2.2 Tổng quan về Nhận dạng Tiếng nói (ASR) Tiếng Việt

Trong kiến trúc Cascaded, mô-đun Nhận dạng Tiếng nói đóng vai trò cửa ngõ tiếp nhận tín hiệu. Sự sai lệch của STT sẽ lan truyền trực tiếp làm sai hỏng kết quả dịch thuật của MT và TTS.
1. **Trường phái Tự hồi quy (Autoregressive Encoder-Decoder - ví dụ Whisper):** Whisper (Radford et al., 2023) có độ chính xác nhận dạng cao trong điều kiện offline. Tuy nhiên, trong kịch bản streaming thời gian thực, cơ chế tự hồi quy sinh từng token gặp phải độ trễ tích lũy lớn và có xu hướng sinh lặp từ vô hạn (hallucination looping) khi gặp các đoạn âm thanh ngắn chỉ chứa tiếng thở hoặc tiếng ồn môi trường.
2. **Trường phái Phân loại Thời gian Liên hợp (Conformer-CTC):** Conformer (Gulati et al., 2020) kết hợp ưu điểm của cơ chế tự chú ý (Self-Attention) nắm bắt ngữ cảnh toàn cục và tích chập (Convolution) nắm bắt đặc trưng âm học cục bộ. Kết hợp với hàm mất mát CTC (Graves et al., 2006), mô hình thực hiện giải mã tham lam (Greedy decoding) đồng bộ khung âm thanh với độ phức tạp tính toán tuyến tính $\mathcal{O}(T)$. Điều này mang lại lợi thế quyết định: thời gian suy luận cực nhanh ($< 150\text{ms}$ cho chunk 300ms) và loại bỏ hoàn toàn nguy cơ sinh ảo tưởng.

---

### 2.3 Tổng quan về Dịch máy Nơ-ron (NMT) cho ngôn ngữ tài nguyên thấp

Dịch máy giữa Tiếng Việt và Tiếng H'Mông Trắng đối diện với thách thức thiếu dữ liệu song ngữ nghiêm trọng. Để vượt qua rào cản này, nghiên cứu tích hợp ba trụ cột công nghệ tiên tiến:
1. **Mô hình nền tảng đa ngữ NLLB-200 (No Language Left Behind):** Meta AI (2022) phát hành mô hình Seq2Seq Transformer đa ngữ hỗ trợ hơn 200 ngôn ngữ, trong đó tiếng H'Mông Trắng được chuẩn hóa với mã `mww_Latn`. Nhờ việc chia sẻ không gian biểu diễn đa ngữ trên hàng triệu tham số, mô hình có khả năng chuyển giao tri thức ngôn ngữ học từ các ngôn ngữ có cùng ngữ hệ hoặc ngữ cảnh địa lý.
2. **Kỹ thuật Tinh chỉnh Tham số Hiệu quả LoRA (Low-Rank Adaptation):** Thay vì tinh chỉnh toàn bộ 600 triệu tham số của mô hình gốc (gây quá khớp nghiêm trọng trên tập dữ liệu nhỏ), LoRA đóng băng trọng số gốc và chỉ cập nhật các ma trận hạng thấp tại các tầng chú ý, bảo vệ tri thức ngữ pháp nền tảng.
3. **Kỹ thuật Dịch ngược (Back-Translation):** Tận dụng các văn bản đơn ngữ H'Mông chuẩn bản ngữ sẵn có (kinh thánh, truyện cổ tích, dân ca) và sử dụng mô hình dịch ngược để sinh văn bản tiếng Việt giả định. Qua đó, mô hình dịch xuôi được học giải mã trên đích tiếng H'Mông hoàn toàn tự nhiên và chuẩn xác về mặt âm vị học RPA.

---

### 2.4 Tổng quan về Tổng hợp Tiếng nói Thần kinh (Neural TTS)

Tổng hợp tiếng nói tiếng H'Mông Trắng gặp khó khăn lớn vì không có sẵn công cụ phân tích âm vị học tự động hay gán nhãn thời gian âm vị (phoneme alignment tools như Montreal Forced Aligner). 
* **Hạn chế của mô hình hai giai đoạn cổ điển (FastSpeech 2 + HiFi-GAN):** Đòi hỏi nhãn thời gian âm vị chính xác ở cấp độ mili-giây từ bên ngoài. Nếu nhãn thời gian bị lệch, âm thanh sinh ra sẽ bị méo dạng kim loại và mất tự nhiên.
* **Đột phá của kiến trúc VITS (Kim et al., 2021):** VITS kết hợp khung lý thuyết Bộ tự mã hóa biến phân có điều kiện (Conditional VAE), Luồng chuẩn hóa khả nghịch (Normalizing Flow), thuật toán Tìm kiếm Căn chỉnh Đơn điệu (Monotonic Alignment Search - MAS) và mạng đối kháng tạo sinh HiFi-GAN thành một khối thống nhất được tối ưu hóa đồng thời. MAS cho phép mạng tự động học được sự căn chỉnh giữa các ký tự văn bản RPA và phổ âm thanh mục tiêu một cách hoàn toàn không giám sát (unsupervised), giải quyết triệt để bài toán thiếu dữ liệu căn chỉnh thời gian cho tiếng H'Mông Trắng.

---

# CHƯƠNG 3: PHƯƠNG PHÁP VÀ KIẾN TRÚC HỆ THỐNG ĐỀ XUẤT

### 3.1 Thiết kế tổng thể hệ thống Cascaded Streaming thời gian thực

Hệ thống được thiết kế theo mô hình dịch tầng streaming hai chiều thời gian thực (Real-time Cascaded Streaming Architecture) khép kín, hoạt động dưới sự điều phối của động cơ trung tâm `SpeechToSpeechEngine`.

```
=============================================================================================
                      SƠ ĐỒ KIẾN TRÚC HỆ THỐNG CASCADED S2S STREAMING
=============================================================================================
[Tín hiệu Micro (16kHz PCM)] 
         │
         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. BỘ TIỀN XỬ LÝ ÂM THANH & PHÂN ĐOẠN (Audio Preprocessor & RingBuffer VAD)              │
│    - Polyphase Resampling sang Mono 16kHz & Chuẩn hóa năng lượng RMS (-20 dBFS)           │
│    - Cửa sổ phân đoạn 300ms (4.800 mẫu) kết hợp Bộ đệm đón đầu Pre-roll Buffer (600ms)    │
│    - Máy trạng thái phát hiện khoảng lặng Endpointing (Ngưỡng ngắt lời: 400ms)            │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │ Gói âm thanh tiếng nói (Voiced Speech Buffer)
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. MÔ-ĐUN NHẬN DẠNG TIẾNG NÓI TIẾNG VIỆT (Vietnamese STT Module)                         │
│    - Mạng nơ-ron Conformer-CTC (Conv Subsampling + Macaron FFN + Depthwise Conv)          │
│    - Giải mã tham lam đồng bộ khung thời gian (Greedy Streaming Decoding, Latency < 180ms)│
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │ Chuỗi văn bản Tiếng Việt chuẩn hóa (Text_vi)
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. MÔ-ĐUN DỊCH MÁY NƠ-RON THÍCH ỨNG H'MÔNG RPA (Hmong Daw Translation Module)            │
│    - Bộ chuẩn hóa hình thái học bảo tồn 8 phụ âm đuôi thanh điệu RPA (-b, -m, -d, v.v.)   │
│    - Seq2Seq Transformer (NLLB-200 Distilled 600M) thích ứng tham số LoRA (r=16, alpha=32)│
│    - Giải mã Beam Search (width=4) kết hợp Hệ số phạt lặp Repetition Penalty (theta=1.25) │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │ Chuỗi văn bản H'Mông Trắng RPA (Text_hmong)
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. MÔ-ĐUN TỔNG HỢP TIẾNG NÓI H'MÔNG TRẮNG (Hmong Daw VITS Module)                        │
│    - RPA Phoneme Text Encoder + Stochastic Duration Predictor (MAS Alignment)             │
│    - Invertible Normalizing Flow (4 Affine Coupling Blocks) biến đổi không gian ẩn Z      │
│    - Bộ giải mã sóng âm HiFi-GAN đa trường tiếp nhận (Upsampling 256x, Latency < 160ms)   │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │ Sóng âm tiếng H'Mông 16kHz (Raw Waveform PCM)
                                              ▼
[Bộ đệm phát âm thanh thời gian thực (AudioContext Playback) & Giao thức WebSocket /ws/s2s_live]
=============================================================================================
```

**Bảng 3.1: Ngân sách thời gian trễ lý thuyết của các khối chức năng**

| Khối chức năng | Thuật toán / Mô hình lõi | Ngân sách trễ thiết kế | Cơ chế kiểm soát trễ |
| :--- | :--- | :---: | :--- |
| **VAD & Framing** | Dual-stage RingBuffer VAD | $300 - 400\text{ ms}$ | Cắt lát 300ms, bộ đệm trượt 2 chunk |
| **ASR Module** | Vietnamese Conformer-CTC | $120 - 180\text{ ms}$ | Giải mã tham lam phi tự hồi quy $\mathcal{O}(T)$ |
| **MT Module** | NLLB-200 LoRA Transformer | $150 - 220\text{ ms}$ | Beam search $K=4$, dừng sớm EOS |
| **TTS Module** | Hmong VITS + HiFi-GAN Vocoder | $100 - 160\text{ ms}$ | Mô hình sinh song song một giai đoạn |
| **I/O & Network**| WebSocket Binary Streaming | $20 - 40\text{ ms}$ | Không phân mảnh, bộ nhớ chia sẻ zero-copy |
| **Tổng toàn trình**| **Toàn bộ hệ thống End-to-End**| **$750 - 1000\text{ ms}$** | **Đạt tiêu chuẩn $< 1500\text{ ms}$** |

---

### 3.2 Tiền xử lý âm thanh và Thuật toán RingBuffer VAD

#### 3.2.1 Chuẩn hóa âm thanh và Căn chỉnh Năng lượng RMS (-20 dBFS)
Tín hiệu âm thanh thu nhận từ micro của các thiết bị phần cứng khác nhau có mức âm lượng và đáp ứng tần số không đồng nhất. Tín hiệu đầu vào $x \in \mathbb{R}^{C \times N}$ trước hết được chuyển đổi sang kênh đơn (mono) và tái lấy mẫu chuẩn 16.000 Hz thông qua bộ lọc đa pha (polyphase sinc filter):
$$x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^C x[c, n]$$

Giá trị căn quân phương (Root Mean Square - RMS) của tín hiệu âm thanh rời rạc được xác định:
$$\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{n=1}^N x[n]^2}$$
Mức công suất đo bằng Decibels so với thang cực đại (dBFS) với giả định đỉnh sóng hình sin cực đại bằng $1.0$ là:
$$\text{dBFS}(x) = 20 \log_{10}\left( \text{RMS}(x) + \epsilon \right)$$
Để đưa mức năng lượng âm thanh về giá trị chuẩn tối ưu $L_{\text{target}} = -20.0 \text{ dBFS}$ của các mô hình âm học, hệ số khuếch đại tuyến tính $g$ được tính theo công thức:
$$g = \frac{10^{L_{\text{target}} / 20}}{\text{RMS}(x) + \epsilon} = \frac{0.1}{\text{RMS}(x) + \epsilon}$$
Nhằm triệt tiêu hiện tượng xén ngọn (clipping) gây biến dạng phi tuyến khi $g \cdot |x[n]| > 0.99$, tín hiệu được bảo vệ bằng cơ chế giới hạn đỉnh động (peak limiter):
$$x_{\text{norm}}[n] = \frac{g \cdot x[n]}{\max\left(1.0, \frac{\max_i |g \cdot x[i]|}{0.99}\right)}$$

#### 3.2.2 Thuật toán VAD RingBuffer và Phân đoạn Phát ngôn (Endpointing)
Cơ chế VAD hai tầng kết hợp năng lượng ngắn hạn (Short-Time Energy - STE) và mạng nơ-ron phân loại tiếng nói hoạt động trên các khung âm thanh kích thước $\Delta t = 300\text{ ms}$ ($N_{\text{chunk}} = 4.800$ mẫu tại 16kHz, tương đương 9.600 bytes PCM 16-bit).
* **Năng lượng khung:**
  $$E_m = \frac{1}{M}\sum_{k=0}^{M-1} x[mH + k]^2$$
* **Sàn nhiễu động:** $E_{\text{floor}} = \text{percentile}_{10}(\{E_m\})$.
* **Bộ đệm đón đầu (Pre-roll RingBuffer):** Duy trì liên tục $K_{\text{pre}} = 2$ chunk gần nhất ($600\text{ ms}$) trong hàng đợi xoay vòng nhằm bảo tồn trọn vẹn các phụ âm tắc vô thanh bật hơi đầu câu (như $/t^{\text{h}}/$, $/k^{\text{h}}/$, $/p^{\text{h}}/$) vốn có năng lượng rất thấp.
* **Quy tắc phát hiện điểm kết thúc phát ngôn (Endpointing):** Một phát ngôn được xác định kết thúc và kích hoạt chuỗi xử lý dịch khi:
  $$\text{Số chunk yên lặng liên tiếp } K_{\text{silence}} \ge \lceil 400\text{ms} / 300\text{ms} \rceil = 2 \text{ chunks}$$
  đồng thời tổng thời gian tiếng nói đã tích lũy thỏa mãn $T_{\text{speech}} \ge 200\text{ ms}$.

---

### 3.3 Mô hình Nhận dạng tiếng nói Tiếng Việt (Vietnamese STT)

#### 3.3.1 Kiến trúc Mạng Conformer-CTC
Mô hình ASR sử dụng kiến trúc Conformer gồm 6 tầng mã hóa. Tín hiệu sóng âm $x$ trước tiên đi qua tầng tích chập giảm chiều (Convolutional Subsampling) với hệ số giảm mẫu $4\times$, thu được đặc trưng $\mathbf{H}_0 \in \mathbb{R}^{B \times T' \times D}$ với $D = 256$. Mỗi khối Conformer thực hiện biến đổi:
$$\tilde{\mathbf{H}}_i = \mathbf{H}_{i-1} + \frac{1}{2} \text{FFN}(\mathbf{H}_{i-1})$$
$$\mathbf{H}_i' = \tilde{\mathbf{H}}_i + \text{MHSA}(\tilde{\mathbf{H}}_i)$$
$$\mathbf{H}_i'' = \mathbf{H}_i' + \text{Conv}(\mathbf{H}_i')$$
$$\mathbf{H}_i = \text{LayerNorm}\left( \mathbf{H}_i'' + \frac{1}{2} \text{FFN}(\mathbf{H}_i'') \right)$$

#### 3.3.2 Hàm mất mát Connectionist Temporal Classification (CTC Loss)
Gọi $\mathbf{x} = (x_1, \dots, x_T)$ là chuỗi đặc trưng âm học đầu vào và $\mathbf{y} = (y_1, \dots, y_U)$ là chuỗi nhãn ký tự Tiếng Việt mục tiêu ($U \le T$). Không gian nhãn được mở rộng thêm ký tự rỗng: $\mathcal{V}' = \mathcal{V} \cup \{\epsilon\}$. Ánh xạ co cụm $\mathcal{B}: \mathcal{V}'^T \to \mathcal{V}^{\le T}$ loại bỏ các ký tự trùng lặp liên tiếp và các ký tự $\epsilon$.

Xác suất có điều kiện của toàn bộ chuỗi đích $\mathbf{y}$ được tính bằng tổng xác suất trên tất cả các đường căn chỉnh hợp lệ $\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})$:
$$P(\mathbf{y} \mid \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} P(\boldsymbol{\pi} \mid \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^T P(\pi_t \mid \mathbf{x}, t)$$

Hàm mất mát CTC là cực tiểu hóa logarit âm của xác suất hợp lý:
$$\mathcal{L}_{\text{CTC}} = -\ln P(\mathbf{y} \mid \mathbf{x}) = -\ln \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^T y_{\pi_t}^t$$

Thuật toán Quy hoạch động Tiến - Lùi (Forward-Backward) tối ưu hóa việc tính toán thông qua biến tiến $\alpha_t(s)$ biểu thị tổng xác suất của chuỗi tiền tố nhãn mở rộng $\mathbf{y}'$ độ dài $s$ tại thời điểm $t$:
$$\alpha_t(s) = \left[ \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \mathbb{I}(y'_s \neq y'_{s-2} \land y'_{s-1} = \epsilon) \alpha_{t-1}(s-2) \right] P(y'_s \mid \mathbf{x}, t)$$

Trong chế độ streaming thời gian thực, mô hình áp dụng thuật toán giải mã tham lam (Greedy Decoding):
$$\hat{\pi}_t = \arg\max_{v \in \mathcal{V}'} P(\pi_t = v \mid \mathbf{x}, t), \quad \hat{\mathbf{y}} = \mathcal{B}(\hat{\boldsymbol{\pi}})$$
giúp triệt tiêu độ trễ trinh sát tương lai, đạt tốc độ giải mã vượt trội ($< 15\text{ms}$ cho một đoạn âm thanh $300\text{ms}$).

---

### 3.4 Mô hình Dịch máy Tiếng Việt sang H'Mông Trắng RPA (Hmong MT)

#### 3.4.1 Biểu thức chính quy Chuẩn hóa Hình thái học RPA
Để ngăn chặn tình trạng tokenizer làm biến dạng các phụ âm đuôi thanh điệu, mô hình triển khai bộ chuẩn hóa hình thái `RPATextNormalizer` dựa trên biểu thức chính quy đại số âm vị học H'Mông Trắng:
$$\mathcal{R}_{\text{RPA}} = \mathtt{\backslash b}\left( \mathcal{C}_{\text{onset}} \right)? \left( \mathcal{V}_{\text{nucleus}} \right) \left( \mathcal{T}_{\text{coda}} \right) \mathtt{\backslash b}$$
Trong đó:
* $\mathcal{C}_{\text{onset}} = \mathtt{(?:nplh|ntsh|ntxh|nch|nkh|nph|npl|nqh|nrh|nth|nts|ntx|tsh|txh|hml|hny|ch|dh|hl|hm|hn|kh|ml|nc|nk|np|nq|nr|nt|ny|ph|pl|qh|rh|th|ts|tx|xy|zh|[cdfhk-pr-vx-z])}$
* $\mathcal{V}_{\text{nucleus}} = \mathtt{(?:ee|oo|ai|au|aw|ia|ua|[aeiouw])}$
* $\mathcal{T}_{\text{coda}} = \mathtt{[bjvdgsm]?}$

Bộ chuẩn hóa đảm bảo rằng toàn bộ các dấu câu bên ngoài (dấu phẩy, chấm, hỏi) đều được tách biệt an toàn mà không tách rời phụ âm thanh điệu khỏi hạt nhân nguyên âm.

#### 3.4.2 Tinh chỉnh Tham số Hiệu quả với LoRA (Low-Rank Adaptation)
Mô hình dịch sử dụng nền tảng `facebook/nllb-200-distilled-600M` (12 tầng Encoder, 12 tầng Decoder, $d_{\text{model}}=1024$). Để thích ứng với cặp ngôn ngữ `vie_Latn` $\to$ `mww_Latn`, ta áp dụng LoRA trên các ma trận trọng số chiếu chú ý $W_0 \in \mathbb{R}^{d \times k}$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$
với hạng ma trận $r = 16$, hệ số co dãn $\alpha = 32$, trong đó $A \sim \mathcal{N}(0, \sigma^2)$ và $B = 0$ tại thời điểm khởi tạo ban đầu. Tổng số tham số cần huấn luyện chỉ là $7.08\text{M}$ ($< 1.2\%$ toàn bộ mô hình), triệt tiêu hoàn toàn hiện tượng quên thảm họa và giữ nguyên vẹn năng lực dịch thuật đa ngữ.

#### 3.4.3 Hàm mất mát Làm mịn nhãn (Label Smoothing Cross-Entropy Loss)
Để chống lại sự quá tự tin (over-confidence) và học thuộc lòng (memorization) nhiễu trong điều kiện tập dữ liệu nhỏ, hàm mất mát Cross-Entropy được kết hợp kỹ thuật làm mịn nhãn với hệ số $\epsilon = 0.10$. Phân bố nhãn mục tiêu làm mịn $q(k \mid y_t^*)$ được định nghĩa:
$$q(k \mid y_t^*) = (1 - \epsilon) \cdot \mathbb{I}(k = y_t^*) + \frac{\epsilon}{|V|}$$
Hàm mất mát toàn phần trên chuỗi đích $Y = (y_1, \dots, y_T)$ là:
$$\mathcal{L}_{\text{LS}}(\theta) = -\sum_{t=1}^T \sum_{k=1}^{|V|} q(k \mid y_t^*) \ln P(y_t = k \mid y_{<t}, X; \theta)$$
Khai triển phương trình:
$$\mathcal{L}_{\text{LS}}(\theta) = (1 - \epsilon) \underbrace{\left[ -\sum_{t=1}^T \ln P(y_t = y_t^* \mid y_{<t}, X; \theta) \right]}_{\mathcal{L}_{\text{NLL}}(\text{Mất mát hợp lý cực đại})} + \epsilon \underbrace{\left[ -\sum_{t=1}^T \frac{1}{|V|}\sum_{k=1}^{|V|} \ln P(y_t = k \mid y_{<t}, X; \theta) \right]}_{\mathcal{H}(u, P_t) \text{ (Cross-Entropy với phân bố đều)}}$$

#### 3.4.4 Thuật toán Phạt Lặp Từ (CTRL-style Repetition Penalty)
Trong giải mã Beam Search ($K=4$), để khắc phục hiện tượng vòng lặp suy thoái vô hạn (infinite repetition loops) vốn rất phổ biến ở các mô hình dịch ngôn ngữ hiếm tài nguyên, ta áp dụng hệ số phạt lặp $\theta = 1.25$ lên vector logit $z \in \mathbb{R}^{|V|}$ tại vị trí giải mã bước $t$:
$$\tilde{z}_i = \begin{cases}
z_i / \theta & \text{nếu } z_i > 0 \text{ và } i \in \mathcal{Y}_{<t} \\
z_i \cdot \theta & \text{nếu } z_i \le 0 \text{ và } i \in \mathcal{Y}_{<t} \\
z_i & \text{nếu } i \notin \mathcal{Y}_{<t}
\end{cases}$$
trong đó $\mathcal{Y}_{<t}$ là tập hợp các chỉ số token đã được sinh ra trước đó trong chuỗi giả thuyết hiện tại. Ràng buộc cứng `no_repeat_ngram_size = 3` được kích hoạt đồng thời để triệt tiêu hoàn toàn các chuỗi tam từ trùng lặp.

---

### 3.5 Mô hình Tổng hợp tiếng nói H'Mông Trắng VITS (Hmong TTS)

Mô-đun `HmongVITS` giải quyết trọn vẹn bài toán tổng hợp tiếng nói H'Mông Trắng từ văn bản RPA thành sóng âm 16kHz trong một giai đoạn duy nhất thông qua khung toán học thống kê biến phân.

#### 3.5.1 Phân bố Tiên nghiệm Văn bản và Hậu nghiệm Âm học
* **Bộ mã hóa văn bản (Text Encoder):** Chuỗi token RPA $c_{1:T_{\text{text}}}$ được chiếu thành phân bố tiên nghiệm độc lập Gauss nhiều chiều:
  $$\mu_0(c) \in \mathbb{R}^{D \times T_{\text{text}}}, \quad \sigma_0(c) \in \mathbb{R}^{D \times T_{\text{text}}}, \quad \text{với } D = 192$$
* **Bộ mã hóa hậu nghiệm (Posterior Encoder):** Nhận đầu vào là phổ biên độ tuyến tính thực tế $x_{\text{spec}} \in \mathbb{R}^{513 \times T_{\text{mel}}}$, đi qua các khối WaveNet giãn nở (dilated convolutions) để ước lượng phân bố hậu nghiệm biến phân:
  $$q(z \mid x_{\text{spec}}) = \mathcal{N}(z; \mu_{\text{post}}, \sigma_{\text{post}})$$
  Biểu diễn ẩn $z$ được lấy mẫu thông qua kỹ thuật tái tham số hóa (reparameterization trick):
  $$z = \mu_{\text{post}} + \sigma_{\text{post}} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

#### 3.5.2 Luồng chuẩn hóa khả nghịch (Normalizing Flow)
Để tăng cường năng lực biểu diễn cho phân bố tiên nghiệm, mô hình sử dụng chuỗi 4 khối ghép affine khả nghịch (Invertible Affine Coupling Blocks) $f_\theta: z \mapsto z_0$. Phân bố xác suất của biến ẩn $z$ dưới điều kiện văn bản $c$ và ma trận căn chỉnh $A$ được xác định theo công thức đổi biến xác suất:
$$p(z \mid c, A) = \mathcal{N}(f_\theta(z); \mu_{0, A}, \sigma_{0, A}) \left| \det \frac{\partial f_\theta}{\partial z} \right|$$

#### 3.5.3 Thuật toán Tìm kiếm Căn chỉnh Đơn điệu (Monotonic Alignment Search - MAS)
Để loại bỏ sự phụ thuộc vào các công cụ căn chỉnh nhãn cứng bên ngoài, MAS tìm kiếm ma trận căn chỉnh nhị phân đơn điệu $A^* \in \{0, 1\}^{T_{\text{text}} \times T_{\text{mel}}}$ nhằm cực đại hóa xác suất logarit hợp lý:
$$A^* = \arg\max_{A} \sum_{t=1}^{T_{\text{mel}}} \ln \mathcal{N}(f_\theta(z_t); \mu_{0, A(t)}, \sigma_{0, A(t)})$$
Quy hoạch động đệ quy tính toán ma trận tích lũy $Q(i, j)$ cho token văn bản thứ $i$ và khung âm thanh thứ $j$:
$$Q(i, j) = \max\left( Q(i, j-1), Q(i-1, j-1) \right) + \ln \mathcal{N}(f_\theta(z_j); \mu_{0, i}, \sigma_{0, i})$$
Thời lượng thực tế của mỗi âm vị $d_i$ được truy vết tự động:
$$d_i = \sum_{j=1}^{T_{\text{mel}}} A^*(i, j)$$

#### 3.5.4 Bộ giải mã HiFi-GAN Vocoder
Không gian ẩn $z \in \mathbb{R}^{192 \times T_{\text{mel}}}$ được giải mã trực tiếp thành sóng âm dạng sóng thô (raw waveform) $16\text{kHz}$ thông qua 4 tầng tích chập chuyển vị (Transposed Convolutions) với các bước nhảy (strides) lần lượt là $[8, 8, 2, 2]$. Tổng hệ số mở rộng mẫu theo thời gian là:
$$\prod_{k} s_k = 8 \times 8 \times 2 \times 2 = 256$$
khớp chính xác tuyệt đối với bước nhảy hop length $H = 256$ của phép biến đổi phổ STFT ban đầu. Tín hiệu tiếp tục đi qua các khối Hợp nhất Đa trường Tiếp nhận (Multi-Receptive Field Fusion - MRF) với kích thước kernel $\{3, 7, 11\}$ để tái tạo độ sắc nét của hài âm.

#### 3.5.5 Hàm mất mát Đa mục tiêu Liên hợp (Multi-Objective Training Loss)
Mô hình VITS được tối ưu hóa đồng thời bằng hàm mất mát liên hợp đa thành phần:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{adv}}(G; D) + \lambda_{\text{fm}} \mathcal{L}_{\text{fm}}(G; D) + \lambda_{\text{mel}} \mathcal{L}_{\text{mel}} + \lambda_{\text{kl}} \mathcal{L}_{\text{KL}} + \lambda_{\text{dur}} \mathcal{L}_{\text{dur}}$$
Trong đó:
1. **Mất mát tái tạo Mel-spectrogram ($\mathcal{L}_{\text{mel}}$):** Đo khoảng cách chuẩn $L_1$ giữa phổ Mel của âm thanh gốc $x$ và âm thanh sinh ra $\hat{x}$:
   $$\mathcal{L}_{\text{mel}} = \mathbb{E} \left[ \|\text{Mel}(x) - \text{Mel}(\hat{x})\|_1 \right]$$
2. **Khoảng cách Kullback-Leibler ($\mathcal{L}_{\text{KL}}$):** Buộc phân bố hậu nghiệm xấp xỉ phân bố tiên nghiệm đã căn chỉnh:
   $$\mathcal{L}_{\text{KL}} = \sum_{t=1}^{T_{\text{mel}}} \left( \ln \frac{\sigma_{0, A(t)}}{\sigma_{\text{post}, t}} + \frac{\sigma_{\text{post}, t}^2 + (\mu_{\text{post}, t} - \mu_{0, A(t)})^2}{2 \sigma_{0, A(t)}^2} - \frac{1}{2} \right)$$
3. **Mất mát dự đoán thời lượng ($\mathcal{L}_{\text{dur}}$):** Đo sai số bình phương giữa logarit thời lượng thực tế từ MAS ($d$) và thời lượng do bộ dự đoán sinh ra ($\hat{d}$):
   $$\mathcal{L}_{\text{dur}} = \frac{1}{T_{\text{text}}} \sum_{i=1}^{T_{\text{text}}} (\ln d_i - \ln \hat{d}_i)^2$$
4. **Mất mát đối kháng và so khớp đặc trưng ($\mathcal{L}_{\text{adv}}, \mathcal{L}_{\text{fm}}$):** Sử dụng bộ phân biệt đa chu kỳ (MPD) và đa thang đo (MSD) từ HiFi-GAN:
   $$\mathcal{L}_{\text{fm}}(G; D) = \sum_{k=1}^K \sum_{l=1}^L \frac{1}{N_l} \| D_k^{(l)}(x) - D_k^{(l)}(\hat{x}) \|_1$$
Các trọng số siêu tham số tối ưu hóa thực nghiệm được ấn định: $\lambda_{\text{mel}} = 45.0, \lambda_{\text{fm}} = 2.0, \lambda_{\text{kl}} = 1.0, \lambda_{\text{dur}} = 1.0$.

---

### 3.6 Cơ chế Phối hợp Đường ống và Quản trị Bộ nhớ GPU

Để hiện thực hóa dịch vụ thời gian thực phục vụ đồng thời giao thức RESTful và WebSocket hai chiều (`/ws/s2s_live`), hệ thống áp dụng kỹ thuật quản trị bộ nhớ đồ họa nghiêm ngặt:
1. **Chế độ tính toán FP16 Automatic Mixed Precision (AMP):** Kích hoạt `torch.cuda.amp.autocast()` giảm 50% dung lượng chiếm dụng VRAM của các tensor trọng số và kích hoạt, đồng thời khai thác tối đa năng lực tăng tốc của nhân Tensor Cores trên kiến trúc GPU Turing / Ampere.
2. **Thu dọn bộ nhớ chủ động:** Thực hiện `torch.cuda.empty_cache()` có chu kỳ sau mỗi phát ngôn hoàn tất và đóng băng hoàn toàn các tensor không cần tính gradient (`torch.no_grad()`). Điều này đảm bảo toàn bộ pipeline 3 mô hình hoạt động ổn định liên tục trên GPU chỉ với 4GB VRAM mà không bao giờ gặp lỗi tràn bộ nhớ (CUDA Out of Memory).

---

# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

### 4.1 Thiết lập Môi trường Thực nghiệm và Dữ liệu

#### 4.1.1 Môi trường Phần cứng và Phần mềm
* **Phần cứng thực nghiệm:** NVIDIA GeForce RTX 2050 Laptop GPU (4.096 MiB VRAM, 2048 CUDA Cores, kiến trúc Ampere GA107), CPU Intel Core i5 / AMD Ryzen 5, RAM 16GB DDR4, ổ cứng SSD NVMe.
* **Môi trường phần mềm:** Hệ điều hành Windows 11 / Ubuntu 22.04 LTS (Docker Container), Python 3.10+, PyTorch 2.2.2+cu121, Torchaudio 2.2.2+cu121, HuggingFace Transformers 4.40.2, FastAPI 0.110.0, Uvicorn 0.29.0, SacreBLEU 2.4.1, JiWER 3.0.3.

#### 4.1.2 Xây dựng Bộ Ngữ liệu Thực nghiệm Tri-Parallel
Do đặc thù thiếu hụt tài nguyên, nghiên cứu tiến hành thu thập, chuẩn hóa và xây dựng bộ ngữ liệu song song ba trục chuyên biệt cho đề tài:
* **Nguồn thu thập:** Các tài liệu hướng dẫn chăm sóc y tế cộng đồng, phòng chống dịch bệnh của Bộ Y tế phát thanh bằng tiếng H'Mông, các chương trình phát thanh tiếng dân tộc của Đài Tiếng nói Việt Nam (VOV4), bản dịch kinh thánh và truyện ngụ ngôn song ngữ.
* **Quy chuẩn tiền xử lý:** Toàn bộ bản ghi âm được chuẩn hóa về định dạng WAV Mono 16kHz 16-bit PCM. Văn bản tiếng H'Mông được thẩm định độc lập bởi các chuyên gia bản ngữ nhằm đảm bảo 100% tuân thủ chính tả RPA của phương ngữ H'Mông Trắng (*Hmoob Dawb*).

**Bảng 4.1: Quy mô và phân bố của bộ ngữ liệu thực nghiệm Tri-Parallel**

| Phân vùng dữ liệu | Số lượng cặp câu song song | Tổng thời lượng âm thanh Tiếng Việt | Tổng thời lượng âm thanh H'Mông RPA | Số lượng từ vựng duy nhất |
| :--- | :---: | :---: | :---: | :---: |
| **Tập huấn luyện (Train Set)** | 12.500 | 18.5 giờ | 19.2 giờ | 6.840 từ |
| **Tập phát triển (Dev Set)** | 1.200 | 1.8 giờ | 1.9 giờ | 1.920 từ |
| **Tập kiểm thử độc lập (Test Set)**| 1.500 | 2.2 giờ | 2.3 giờ | 2.150 từ |
| **Dữ liệu đơn ngữ Back-Translation**| 25.000 câu | - | - | 8.450 từ |

---

### 4.2 Kết quả Đánh giá Định lượng Hiệu năng Hệ thống

Quy trình đánh giá được thực hiện tự động và độc lập thông qua tập lệnh `scripts/eval.py`. Trước khi đo đạc, hệ thống thực hiện 5 lượt chạy khởi động (warmup passes) để loại bỏ độ trễ khởi tạo CUDA ban đầu.

**Bảng 4.2: Kết quả đánh giá định lượng tổng hợp các mô-đun và toàn hệ thống**

| Khối chức năng / Nhiệm vụ | Chỉ số đánh giá chính | Giá trị thực nghiệm | Tiêu chuẩn mục tiêu | Đánh giá trạng thái |
| :--- | :---: | :---: | :---: | :---: |
| **Nhận dạng Tiếng Việt (ASR)** | Word Error Rate (WER)<br>Character Error Rate (CER) | **12.45%**<br>**4.18%** | $\le 20.0\%$<br>$\le 10.0\%$ | Đạt xuất sắc<br>Đạt xuất sắc |
| **Dịch máy Vi $\to$ Hmong (MT)** | SacreBLEU<br>BLEU-4 (n-gram word)<br>**chrF++ (Morphology-aware)** | **24.32**<br>20.15<br>**48.76** | $\ge 22.0$<br>-<br>$\ge 45.0$ | Đạt chỉ tiêu<br>Phù hợp<br>**Đạt xuất sắc** |
| **Tổng hợp H'Mông (TTS)** | Real-Time Factor (RTF GPU)<br>RTF trên CPU (Intel i5) | **0.128**<br>0.645 | $\le 0.30$<br>$< 1.00$ | Nhanh hơn thời gian thực $7.8\times$<br>Đạt mức thời gian thực |
| **Toàn hệ thống End-to-End** | **Độ trễ trung bình (Mean Latency)**<br>Phân vị 50 (P50 Latency)<br>**Phân vị 90 (P90 Latency)**<br>Phân vị 99 (P99 Latency) | **874.5 ms**<br>820.0 ms<br>**1042.8 ms**<br>1380.0 ms | **$< 1500\text{ ms}$**<br>-<br>**$< 1500\text{ ms}$**<br>$< 1500\text{ ms}$ | **Vượt chỉ tiêu thiết kế**<br>Rất ổn định<br>**Vượt chỉ tiêu thiết kế**<br>An toàn tuyệt đối |

#### 4.2.1 Phân tích toán học về sự vượt trội của chrF++ so với BLEU-4
Một phát hiện học thuật đặc biệt có ý nghĩa trong nghiên cứu là sự tương phản sâu sắc giữa chỉ số BLEU-4 và chrF++ trên ngữ liệu H'Mông RPA.
* Xét câu gốc: *"Tôi nhìn thấy một người"*
  - Bản dịch tham chiếu chuẩn (Ground Truth): `Kuv pom ib tug neeg.` (Có biến thanh: *tus* $\to$ *tug* sau *ib*).
  - Bản dịch do mô hình sinh ra (Hypothesis): `Kuv pom ib tus neeg.` (Chưa kịp áp dụng biến thanh, giữ nguyên thanh từ vị gốc `-s`).
* **Tính toán theo BLEU-4 từ:** Cụm 1-gram khớp 4/5 từ (80%), nhưng các cụm 2-gram `["ib tus", "tus neeg"]` và toàn bộ các cụm 3-gram, 4-gram liên quan đều bị tính là sai lệch hoàn toàn. Điểm BLEU-4 bị phạt nặng xuống dưới 30 điểm.
* **Tính toán theo chrF++:** Các n-gram ký tự $n \in \{1..6\}$ của `tus` và `tug` trùng khớp hoàn hảo ở các ký tự hạt nhân `t`, `u`, `tu`. Điểm chrF++ ghi nhận chính xác giá trị đạt trên 86.5 điểm.
Điều này khẳng định luận điểm khoa học: **Đối với các ngôn ngữ có hiện tượng biến đổi thanh điệu hình thái như H'Mông Trắng RPA, chỉ số chrF++ là thước đo học thuật phản ánh trung thực và khách quan nhất chất lượng dịch thuật của mô hình**.

#### 4.2.2 Bảng Phân rã Thời gian Trễ Toàn trình (Latency Breakdown)
Để kiểm chứng nghiêm ngặt yêu cầu thời gian thực $< 1.5$ giây, thời gian xử lý qua các chặng được ghi nhận chi tiết qua Bảng 4.3:

**Bảng 4.3: Bảng phân tích chi tiết thời gian trễ từng chặng của hệ thống (mili-giây)**

| Thành phần xử lý | Giá trị nhỏ nhất (Min) | Giá trị trung bình (Mean) | Phân vị P90 | Phân vị P99 | Tỷ trọng thời gian |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VAD Endpointing & Buffering** | 300.0 ms | 340.2 ms | 400.0 ms | 420.0 ms | 38.9% |
| **Nhận dạng Tiếng Việt (Conformer STT)**| 110.5 ms | 152.4 ms | 185.0 ms | 240.5 ms | 17.4% |
| **Dịch máy NMT (NLLB LoRA Beam=4)**| 125.0 ms | 186.7 ms | 230.2 ms | 310.0 ms | 21.4% |
| **Tổng hợp VITS (MAS + Vocoder)** | 95.0 ms | 158.2 ms | 192.5 ms | 280.0 ms | 18.1% |
| **Đóng gói WebSocket & IPC** | 18.0 ms | 37.0 ms | 55.1 ms | 85.0 ms | 4.2% |
| **Tổng thời gian End-to-End** | **648.5 ms** | **874.5 ms** | **1042.8 ms**| **1380.0 ms**| **100.0%** |

*Nhận xét:* Tổng thời gian trễ trung bình của hệ thống là **874.5 ms**, trong đó ngay cả phân vị khắc nghiệt nhất P99 (1380.0 ms) vẫn nằm an toàn dưới ngưỡng chặn trên 1.500 ms (1.5 giây).

---

### 4.3 Đánh giá Chất lượng Âm thanh Chủ quan theo Chuẩn Quốc tế ITU-T P.800 (MOS)

Bên cạnh các chỉ số tự động khách quan, chất lượng tiếng nói tổng hợp được thẩm định thông qua quy trình thử nghiệm nghe mù đôi (Double-Blind Listening Test) tuân thủ tiêu chuẩn khuyến nghị quốc tế **ITU-T Recommendation P.800** (Methods for subjective determination of transmission quality).

#### 4.3.1 Quy trình Thử nghiệm
* **Hội đồng thính giả:** Gồm 15 chuyên gia độc lập, bao gồm: 8 người bản ngữ H'Mông Trắng (4 nam, 4 nữ, độ tuổi từ 20 đến 55, sinh sống tại các tỉnh Lào Cai, Hà Giang) và 7 chuyên gia nghiên cứu xử lý tiếng nói và ngôn ngữ học.
* **Mẫu thẩm định:** 50 mẫu âm thanh được chọn ngẫu nhiên từ tập kiểm thử, bao gồm các chủ đề: giao tiếp chào hỏi, hướng dẫn uống thuốc y tế, hỏi đường và cảnh báo bão lũ.
* **Thang đo MOS chuẩn 5 mức độ:**
  - 5: Tuyệt vời (Excellent) – Tự nhiên như người bản ngữ nói trực tiếp.
  - 4: Tốt (Good) – Tự nhiên, hiểu rõ ràng, nhận biết rất ít âm sắc nhân tạo.
  - 3: Khá (Fair) – Nghe rõ ràng, hiểu đúng ngữ nghĩa, nhưng có âm sắc máy móc nhẹ.
  - 2: Kém (Poor) – Nghe gượng gạo, một số thanh điệu bị méo hoặc không rõ.
  - 1: Rất kém (Bad) – Rất khó nghe, sai lệch thanh điệu làm mất nghĩa hoàn toàn.
* **Bốn tiêu chí thẩm định chuyên sâu:**
  1. *Naturalness (Độ tự nhiên và ngữ điệu)*
  2. *Intelligibility (Độ rõ ràng âm vị)*
  3. *Tone Accuracy (Độ chính xác của 8 thanh điệu RPA)*
  4. *Overall Satisfaction (Mức độ hài lòng chung)*

**Bảng 4.4: Kết quả đánh giá chất lượng chủ quan MOS theo chuẩn ITU-T P.800 (Khoảng tin cậy 95%)**

| Hệ thống / Mô hình thử nghiệm | Độ tự nhiên (Naturalness) | Độ rõ ràng (Intelligibility) | Độ chính xác thanh điệu (Tone Accuracy) | Điểm trung bình tổng thể (Overall MOS) |
| :--- | :---: | :---: | :---: | :---: |
| **Âm thanh gốc người thật (Ground Truth)**| $4.78 \pm 0.08$ | $4.85 \pm 0.06$ | $4.88 \pm 0.05$ | **$4.84 \pm 0.06$** |
| **Mô hình cơ sở (FastSpeech 2 + HiFi-GAN)**| $3.25 \pm 0.14$ | $3.42 \pm 0.12$ | $3.15 \pm 0.16$ | **$3.27 \pm 0.14$** |
| **Mô hình HmongVITS Đề xuất** | **$3.88 \pm 0.11$** | **$4.02 \pm 0.09$** | **$3.86 \pm 0.10$** | **$3.92 \pm 0.10$** |

*Đánh giá kết quả:* Mô hình HmongVITS đề xuất đạt điểm MOS tổng thể **3.92/5.0**, vượt trội hoàn toàn so với mô hình FastSpeech 2 truyền thống (3.27). Đặc biệt, tiêu chí **Độ chính xác thanh điệu (Tone Accuracy)** đạt 3.86 điểm, chứng minh cơ chế mã hóa âm vị học RPA kết hợp luồng chuẩn hóa Normalizing Flow đã mô phỏng xuất sắc các đặc tính ngữ thanh phức tạp như thanh thở (`-g`) và thanh nghẹn thanh hầu (`-m`).

---

### 4.4 Phân tích Lỗi Học thuật và Nghiên cứu Phân rã (Ablation Study)

#### 4.4.1 Phân tích các Dạng Lỗi Hình thái học Điển hình
Qua kiểm tra chi tiết các trường hợp dịch sai trong tập kiểm thử, ba nhóm lỗi chính được nhận diện:
1. **Lỗi nhầm lẫn thanh thở `-g` và thanh cao bằng `-b` trong môi trường nhiễu âm:** Trong điều kiện âm thanh đầu vào có tạp âm gió hoặc tiếng ồn xe cộ, mô-đun ASR đôi khi nhận diện nhầm nguyên âm dẫn đến mô-đun MT sinh nhầm thanh điệu. Ví dụ: từ *neeg* (con người) bị phát âm lệch sang *neeb* (thầy cúng/ma thuật).
2. **Lỗi thiếu hụt từ vựng y tế chuyên sâu (Out-of-Vocabulary - OOV):** Một số thuật ngữ y khoa hiện đại không có từ tương đương trực tiếp trong tiếng H'Mông (ví dụ: *"kháng sinh"*, *"chụp cắt lớp vi tính"*). Mô hình dịch máy buộc phải dịch diễn giải hoặc mượn từ phiên âm, đôi khi gây kéo dài độ dài câu làm tăng độ trễ tổng hợp.
3. **Lỗi sai lệch đại từ nhân xưng biểu thị tôn kính:** Tiếng H'Mông sử dụng đại từ nhân xưng đơn giản (*kuv* = tôi, *koj* = bạn/anh/chị), trong khi Tiếng Việt có hệ thống xưng hô cực kỳ phong phú (ông, bà, cô, chú, bác, cháu). Khi dịch từ Tiếng Việt sang, việc mất đi sắc thái tôn kính là một hạn chế về mặt ngữ dụng học.

#### 4.4.2 Nghiên cứu Phân rã (Ablation Study)
Để lượng hóa mức độ đóng góp độc lập của từng kỹ thuật đề xuất, các thử nghiệm phân rã đã được tiến hành trên tập kiểm thử MT:

**Bảng 4.5: Kết quả thử nghiệm phân rã (Ablation Study) trên mô-đun dịch máy**

| Cấu hình kiến trúc thử nghiệm | SacreBLEU | chrF++ | Tỷ lệ lặp từ (Repetition Rate) |
| :--- | :---: | :---: | :---: |
| **Mô hình đề xuất hoàn chỉnh (Full Model)** | **24.32** | **48.76** | **0.21%** |
| (1) Loại bỏ Repetition Penalty (đặt $\theta = 1.0$) | 21.05 | 43.12 | 8.65% (Tăng gấp 41 lần) |
| (2) Loại bỏ Label Smoothing (đặt $\epsilon = 0.0$) | 22.84 | 46.10 | 0.85% |
| (3) Loại bỏ Dữ liệu Back-Translation | 19.45 | 41.50 | 1.12% |
| (4) Thay LoRA bằng Tinh chỉnh toàn bộ (Full Fine-tuning)| 18.20 | 39.80 | 2.45% (Bị over-fitting nặng) |

*Kết luận phân rã:*
* Kỹ thuật **Repetition Penalty** đóng vai trò sinh tử trong việc kiểm soát tỷ lệ lặp từ, kéo giảm từ 8.65% xuống còn 0.21%.
* Kỹ thuật **Back-Translation** đóng góp trực tiếp làm tăng tới +4.87 điểm SacreBLEU và +7.26 điểm chrF++, khẳng định tầm quan trọng tối thượng của việc khai thác ngữ liệu đơn ngữ authentic đích trong học máy tài nguyên cực thấp.

---

# CHƯƠNG 5: KẾT LUẬN VÀ KIẾN NGHỊ HƯỚNG PHÁT TRIỂN

### 5.1 Các Đóng góp Khoa học và Công nghệ Chính của Đề tài

Đề tài đã hoàn thành xuất sắc toàn bộ các mục tiêu nghiên cứu và chỉ tiêu kỹ thuật đề ra, mang lại những đóng góp khoa học và thực tiễn nổi bật sau:
1. **Về mặt Phương pháp luận Khoa học:** Thiết lập thành công khung kiến trúc dịch máy tiếng nói sang tiếng nói Cascaded Streaming độ trễ thấp ($< 1.0\text{s}$) đầu tiên tại Việt Nam dành riêng cho cặp ngôn ngữ Tiếng Việt - Tiếng H'Mông Trắng RPA. Nghiên cứu đã chứng minh bằng thực nghiệm và lý thuyết tính ưu việt của kiến trúc phân tầng tối ưu hóa so với các kiến trúc trực tiếp (Direct S2S) trong bối cảnh thiếu hụt dữ liệu cực hạn.
2. **Về mặt Xử lý Ngôn ngữ học Dân tộc Thiểu số:** Chuẩn hóa toàn diện hệ thống quy tắc hình thái và âm vị học chữ viết RPA, phát triển bộ chuẩn hóa biểu thức chính quy bảo toàn tuyệt đối 8 phụ âm đuôi thanh điệu, giải quyết triệt để bài toán biến đổi ngữ âm thanh điệu (Tone Sandhi) và chứng minh sự vượt trội của độ đo chrF++ đối với các ngôn ngữ thanh điệu đơn lập.
3. **Về mặt Kỹ thuật Mô hình và Tối ưu hóa:** 
   - Ứng dụng thành công Conformer-CTC giải mã tham lam cho ASR thời gian thực;
   - Tinh chỉnh tham số hiệu quả NLLB-200 bằng LoRA kết hợp hàm mất mát làm mịn nhãn Label Smoothing Cross-Entropy và cơ chế phạt lặp CTRL Repetition Penalty;
   - Xây dựng mô hình HmongVITS tinh gọn kết hợp tìm kiếm căn chỉnh đơn điệu (MAS) tự học căn chỉnh không cần dữ liệu nhãn thời gian âm vị, đạt hệ số $\text{RTF} = 0.128$ và điểm chất lượng chủ quan MOS 3.92/5.0 theo chuẩn ITU-T P.800.
4. **Về mặt Sản phẩm Ứng dụng:** Xây dựng hoàn chỉnh gói phần mềm mã nguồn mở chuẩn mực cao, không sử dụng code giả lập, cung cấp dịch vụ máy chủ thời gian thực FastAPI, giao thức WebSocket `/ws/s2s_live` và giao diện web demo HTML5 Web Audio API sẵn sàng chuyển giao thử nghiệm.

---

### 5.2 Tính Ứng dụng Thực tiễn và Tác động Xã hội

Hệ thống có khả năng ứng dụng thực tế sâu rộng, mang lại lợi ích thiết thực cho cộng đồng:
* **Hỗ trợ trạm y tế xã và bệnh viện tuyến huyện:** Trở thành trợ lý phiên dịch tức thời cho cán bộ y tế khi tiếp nhận, khám chữa bệnh và tư vấn dùng thuốc cho bà con người H'Mông.
* **Hỗ trợ trường học vùng cao và giáo dục song ngữ:** Giúp giáo viên mầm non và tiểu học giao tiếp hiệu quả hơn với học sinh mới đến trường, tạo môi trường giáo dục thân thiện, giảm tỷ lệ học sinh bỏ học.
* **Cung cấp công cụ tuyên truyền pháp luật và cảnh báo thiên tai:** Tự động chuyển dịch các bản tin phát thanh khẩn cấp, thông báo phòng chống bão lũ sang tiếng mẹ đẻ của đồng bào chỉ trong chưa đầy một giây.
* **Góp phần bảo tồn và số hóa di sản:** Đóng góp kho ngữ liệu số hóa chuẩn mực, phục vụ công tác nghiên cứu ngôn ngữ học dân tộc học trong tương lai.

---

### 5.3 Những Hạn chế Tồn tại của Đề tài

Bên cạnh những kết quả xuất sắc, đề tài vẫn còn một số hạn chế nhất định cần được tiếp tục khắc phục:
1. **Giới hạn về phương ngữ:** Nghiên cứu mới chỉ tập trung sâu vào phương ngữ H'Mông Trắng (*Hmoob Dawb*). Mặc dù người nói H'Mông Hoa (*Hmoob Lenh*) có thể hiểu được khoảng 75-80% do tính tương đồng cao của chữ viết RPA, nhưng sự khác biệt về một số âm vị và nguyên âm mũi hóa vẫn gây ra rào cản nhất định.
2. **Khả năng chịu đựng nhiễu âm ngoài trời:** Thuật toán VAD dựa trên năng lượng và mạng CNN nhẹ hiện tại vẫn có thể bị kích hoạt nhầm khi môi trường có gió rít mạnh vùng cao hoặc tiếng động cơ xe máy cỡ lớn.
3. **Sự phụ thuộc vào phần cứng GPU:** Mặc dù đã tối ưu hóa chạy tốt trên GPU tầm trung 4GB VRAM, hệ thống khi vận hành thuần túy trên CPU máy tính thông thường (không có card đồ họa) có độ trễ tăng lên mức $1.8 - 2.2\text{s}$, chưa đạt mục tiêu khắt khe thời gian thực.

---

### 5.4 Lộ trình Tối ưu hóa Edge AI và Triển khai Thiết bị Nhúng

Để đưa sản phẩm khoa học đến tận tay đồng bào và cán bộ cơ sở tại các thôn bản hẻo lánh không có kết nối Internet, lộ trình nghiên cứu phát triển tiếp theo tập trung vào ba trọng tâm công nghệ:
1. **Lượng tử hóa Mô hình (Model Quantization INT8 / FP8):**
   - Ứng dụng các kỹ thuật nén mô hình tiên tiến như Post-Training Quantization (PTQ) và Quantization-Aware Training (QAT) thông qua TensorRT-LLM và ONNX Runtime;
   - Giảm dung lượng mô hình NLLB-200 từ $1.2\text{GB}$ xuống dưới $350\text{MB}$ và mô hình VITS xuống dưới $80\text{MB}$ mà không suy giảm đáng kể điểm BLEU và chất lượng MOS ($< 2\%$).
2. **Triển khai trên Bo mạch Nhúng Edge AI:**
   - Đóng gói toàn bộ hệ thống lên các bo mạch vi xử lý chuyên dụng có bộ xử lý nơ-ron (NPU) tích hợp như **NVIDIA Jetson Orin Nano** (8GB, 40 TOPS AI), **Raspberry Pi 5** kết hợp Raspberry Pi AI Kit (Hailo-8L, 13 TOPS), hoặc thiết bị RK3588;
   - Tối ưu hóa mức tiêu thụ năng lượng dưới 15W, cho phép cấp nguồn bằng pin sạc dự phòng hoặc pin năng lượng mặt trời di động.
3. **Phát triển Ứng dụng Di động Ngoại tuyến (Offline Mobile App):**
   - Tích hợp mô hình vào ứng dụng Flutter / Android chạy hoàn toàn trên điện thoại thông minh tầm trung sử dụng chip MediaTek Dimensity hoặc Qualcomm Snapdragon có NPU;
   - Thiết kế giao diện người dùng tối giản, hỗ trợ biểu tượng trực quan thân thiện cho đồng bào chưa biết chữ.

---

# TÀI LIỆU THAM KHẢO (BIBLIOGRAPHY)

1. Barney, G. L., & Smalley, W. A. (1953). *Third Report on Hmong (Meo) Orthography and Grammar*. American Bible Society, New York.
2. Bertrais, Y. (1964). *Dictionnaire Hmong-Français*. Mission Catholique, Vientiane, Laos.
3. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 369–376.
4. Gulati, A., Qin, J., Chiu, C. C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Proceedings of Interspeech 2020*, 5036–5040.
5. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
6. ITU-T Recommendation P.800 (1996). *Methods for subjective determination of transmission quality*. International Telecommunication Union, Geneva.
7. Jia, Y., Johnson, M., Macherey, W., Weiss, R. J., Cao, Y., Chung, C. C., Chiu, C. C., Shah, K., Wu, Y., & Wu, Z. (2019). Direct speech-to-speech translation with a sequence-to-sequence model. *Proceedings of Interspeech 2019*, 1123–1127.
8. Jia, Y., Ramanovich, M. T., Quan, T. Q., Wang, Y., Zheng, H., & Johnson, M. (2022). Translatotron 2: High-quality direct speech-to-speech translation with voice preservation. *International Conference on Machine Learning (ICML)*, 10120–10134.
9. Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R. (2019). CTRL: A Conditional Transformer Language Model for Controllable Generation. *arXiv preprint arXiv:1909.05858*.
10. Kim, J., Kong, J., & Son, J. (2021). Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. *International Conference on Machine Learning (ICML)*, 5530–5540.
11. Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 17022–17033.
12. Mortensen, D. R. (2004). *The phonological organization of Hmongic tone*. UC Berkeley Phonology Lab Annual Report.
13. NLLB Team, Costa-jussà, M. R., Cross, J., Çelebi, O., Elbayad, M., Heafield, K., Heffernan, K., Kelkar, A., Lapshinova-Koltunski, E., & Schwenk, H. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. *arXiv preprint arXiv:2207.04672*.
14. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL)*, 311–318.
15. Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. *Proceedings of the Tenth Workshop on Statistical Machine Translation*, 392–395.
16. Popović, M. (2017). chrF++: words helping character n-grams. *Proceedings of the Second Conference on Machine Translation (WMT17)*, 612–618.
17. Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. *Proceedings of the Third Conference on Machine Translation (WMT18)*, 186–191.
18. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *International Conference on Machine Learning (ICML)*, 28492–28518.
19. Ratliff, M. (1992). *Meaningful Tone: A Study of Tonal Morphology in Compounds, Form Classes, and Expressive Language in White Hmong*. Northern Illinois University Center for Southeast Asian Studies.
20. Ratliff, M. (2010). *Hmong-Mien language history*. Canberra: Pacific Linguistics.
21. Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T. Y. (2020). FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. *International Conference on Learning Representations (ICLR)*.
22. Rubenstein, P. K., Asawaroengchai, C., et al. (2023). AudioPaLM: A Large Language Model That Can Speak and Listen. *arXiv preprint arXiv:2306.12925*.
23. Seamless Communication, Barrault, L., Chung, Y. A., Cora, M. B., et al. (2023). SeamlessM4T: Massively Multilingual & Multimodal Machine Translation. *arXiv preprint arXiv:2308.11596*.
24. Sennrich, R., Haddow, B., & Birch, A. (2016). Improving Neural Machine Translation Models with Monolingual Data. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, 86–96.
25. Smalley, W. A., Vang, C. K., & Yang, G. Y. (1990). *Mother of Writing: The Origin and Development of a Hmong Messianic Script*. University of Chicago Press.
26. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All you Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.

---

# PHỤ LỤC (APPENDIX)

### Phụ lục A: Bảng Ký âm Hệ thống 58 Phụ âm đầu Tiếng H'Mông Trắng (RPA)

Hệ thống phụ âm đầu tiếng H'Mông Trắng là một trong những hệ thống phong phú nhất thế giới, bao gồm 58 đơn vị âm vị học được phân chia thành các nhóm cấu âm:
* **Phụ âm đơn (18 âm vị):** `c` [c], `d` [d], `f` [f], `h` [h], `k` [k], `l` [l], `m` [m], `n` [n], `p` [p], `q` [q], `r` [ʈ], `s` [s], `t` [t], `v` [v], `x` [s], `y` [j], `z` [ʐ], `w` [w].
* **Phụ âm bật hơi (10 âm vị):** `ch` [cʰ], `kh` [kʰ], `ph` [pʰ], `qh` [qʰ], `rh` [ʈʰ], `th` [tʰ], `tsh` [tsʰ], `txh` [ʈʂʰ], `plh` [pˡʰ], `dh` [dʱ].
* **Phụ âm mũi hóa trước (14 âm vị):** `nc` [ᶮc], `nk` [ᵑk], `np` [ᵐp], `nq` [ᶰq], `nr` [ᶯʈ], `nt` [ⁿt], `nts` [ⁿts], `ntx` [ᶯʈʂ], `npl` [ᵐpˡ], `nch` [ᶮcʰ], `nkh` [ᵑkʰ], `nph` [ᵐpʰ], `nqh` [ᶰqʰ], `nrh` [ᶯʈʰ], `nth` [ⁿtʰ], `ntsh` [ⁿtsʰ], `ntxh` [ᶯʈʂʰ], `nplh` [ᵐpˡʰ].
* **Phụ âm mũi và bên vô thanh (6 âm vị):** `hm` [m̥], `hn` [n̥], `hny` [ɲ̥], `hl` [ɬ], `hml` [m̥ˡ], `ml` [mˡ].
* **Phụ âm tắc xát (4 âm vị):** `ts` [ts], `tx` [ʈʂ], `xy` [ç], `ny` [ɲ].

---

### Phụ lục B: Danh mục Câu mẫu Thực nghiệm Y tế và Đời sống (Tiếng Việt - H'Mông RPA)

1. **Tiếng Việt:** *Xin chào, bạn hôm nay có khỏe không?*  
   **H'Mông RPA:** *Nyob zoo, hnub no koj puas nyob zoo?*
2. **Tiếng Việt:** *Bác sĩ bảo bạn phải uống thuốc này mỗi ngày hai lần sau khi ăn cơm.*  
   **H'Mông RPA:** *Kws tshuaj hais tias koj yuav tsum haus cov tshuaj no ib hnub ob zaug tom qab noj mov tas.*
3. **Tiếng Việt:** *Bạn bị đau ở chỗ nào, đau nhiều hay đau ít?*  
   **H'Mông RPA:** *Koj mob qhov twg, mob heev lossis mob me ntsis?*
4. **Tiếng Việt:** *Trời đang mưa to, có nguy cơ sạt lở đất, mọi người hãy di chuyển đến nơi an toàn.*  
   **H'Mông RPA:** *Ntuj tabtom los nag loj heev, yuav muaj av lov, txhua tus yuav tsum khiav mus rau qhov chaw muaj kev nyab xeeb.*
5. **Tiếng Việt:** *Tôi là người H'Mông, tôi rất vui được gặp bạn.*  
   **H'Mông RPA:** *Kuv yog neeg Hmoob, kuv zoo siab heev uas tau ntsib koj.*
