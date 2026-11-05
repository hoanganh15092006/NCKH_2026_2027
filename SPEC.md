# ĐẶC TẢ HỆ THỐNG & TÍNH NĂNG (SYSTEM & FEATURE SPECIFICATION)

- **Tên hệ thống:** Hệ thống Dịch máy Speech-to-Speech Tiếng Việt - Tiếng H'Mông Trắng (RPA)
- **Mục tiêu:** Cung cấp dịch vụ dịch tiếng nói hai chiều thời gian thực end-to-end với độ trễ < 1.5s, nhận dạng chính xác tiếng Việt thực tế từ microphone/tệp âm thanh, dịch chuẩn ngữ nghĩa sang chữ viết H'Mông RPA và phát âm thanh H'Mông tự nhiên.
- **Trạng thái:** Đã hoàn thành core modules và test suite; đang hoàn thiện tài liệu chuẩn hóa và CI/CD.

---

## 1. GIAO KÈO ĐẦU VÀO VÀ ĐẦU RA (INPUT / OUTPUT CONTRACT)

### 1.1 Endpoint Nhận dạng Tiếng nói (STT): `POST /api/v1/stt`
- **Đầu vào (Input):**
  - `file`: Multipart file upload định dạng WAV/MP3/FLAC/OGG, sample rate bất kỳ (hệ thống tự chuẩn hóa sang 16kHz mono).
- **Đầu ra (Output):**
  - **Trường hợp thành công (Success - HTTP 200 OK):**
    ```json
    {
      "success": true,
      "text": "xin chào bà con",
      "duration_sec": 1.45,
      "latency_ms": 38.2
    }
    ```
  - **Trường hợp lỗi (Error - HTTP 400 / 422):**
    ```json
    {
      "success": false,
      "error": {
        "code": "INVALID_AUDIO_FORMAT",
        "message": "Tệp âm thanh không hợp lệ hoặc rỗng."
      }
    }
    ```

### 1.2 Endpoint Dịch máy (MT): `POST /api/v1/translate`
- **Đầu vào (Input):**
  - Body JSON:
    ```json
    {
      "text": "xin chào bà con",
      "src_lang": "vie_Latn",
      "tgt_lang": "mww_Latn"
    }
    ```
- **Đầu ra (Output):**
  - **Trường hợp thành công (Success - HTTP 200 OK):**
    ```json
    {
      "success": true,
      "src_text": "xin chào bà con",
      "translated_text": "nyob zoo cov kwv tij",
      "src_lang": "vie_Latn",
      "tgt_lang": "mww_Latn",
      "latency_ms": 1.2
    }
    ```
  - **Trường hợp lỗi (Error - HTTP 400 / 422):**
    ```json
    {
      "success": false,
      "error": {
        "code": "INVALID_INPUT_TEXT",
        "message": "Văn bản đầu vào rỗng hoặc không hợp lệ."
      }
    }
    ```

### 1.3 Endpoint Tổng hợp Tiếng nói (TTS): `POST /api/v1/tts`
- **Đầu vào (Input):**
  - Body JSON:
    ```json
    {
      "text": "nyob zoo cov kwv tij",
      "output_format": "wav"
    }
    ```
- **Đầu ra (Output):**
  - **Trường hợp thành công (Success - HTTP 200 OK):** Binary stream `audio/wav`, 16,000 Hz, 16-bit Mono PCM.
  - **Trường hợp lỗi (Error - HTTP 400 / 422):**
    ```json
    {
      "success": false,
      "error": {
        "code": "SYNTHESIS_ERROR",
        "message": "Không thể tổng hợp giọng đọc từ văn bản rỗng."
      }
    }
    ```

### 1.4 Endpoint Dịch Tiếng nói Trọn gói (S2S): `POST /api/v1/s2s`
- **Đầu vào (Input):**
  - `file`: Multipart audio file tiếng Việt (WAV/MP3).
- **Đầu ra (Output):**
  - **Trường hợp thành công (Success - HTTP 200 OK):** Binary stream `audio/wav` âm thanh tiếng H'Mông tổng hợp, kèm metadata trong response headers:
    - `X-Transcript-Vi`: Chuỗi nhận dạng tiếng Việt (URL encoded hoặc ASCII safe).
    - `X-Translation-Hmong`: Chuỗi dịch tiếng H'Mông RPA.
    - `X-Latency-Ms`: Thời gian xử lý end-to-end (ms).
  - **Trường hợp lỗi (Error - HTTP 400 / 422):**
    ```json
    {
      "success": false,
      "error": {
        "code": "S2S_PROCESSING_ERROR",
        "message": "Xử lý chuỗi S2S thất bại."
      }
    }
    ```

### 1.5 WebSocket Streaming Thời gian thực: `WS /ws/s2s_live`
- **Đầu vào (Input):** Stream nhị phân các mẩu PCM 16-bit 16kHz (mỗi chunk 300ms = 9,600 bytes).
- **Đầu ra (Output):** Stream các chunk âm thanh tiếng H'Mông được sinh lập tức khi VAD phát hiện kết thúc câu nói (endpointing), kèm sự kiện JSON qua kênh WebSocket, đảm bảo độ trễ end-to-end < 1.5s.

### 1.6 Endpoint Kiểm tra Trạng thái (Health Check): `GET /api/v1/health`
- **Đầu ra (Output - HTTP 200 OK):**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "models": {
      "stt": "loaded",
      "mt": "loaded",
      "tts": "loaded"
    }
  }
  ```

---

## 2. CÁC TRƯỜNG HỢP BIÊN BẮT BUỘC XỬ LÝ (EDGE CASES)

*Mọi module và endpoint bắt buộc phải xử lý an toàn các tình huống sau:*

1. **Khoảng lặng tuyệt đối hoặc nhiễu nền:**
   - Âm thanh đầu vào toàn số 0 hoặc biên độ < threshold VAD ➔ VAD lọc bỏ, STT trả về chuỗi rỗng `""`, không kích hoạt forward pass MT/TTS, không gây lỗi chia cho 0.
2. **Âm thanh méo biên độ cực đại (Acoustic Clipping):**
   - Biên độ vượt ngưỡng > 0 dBFS (+6 dBFS đến +12 dBFS) ➔ Bộ tiền xử lý tự động chuẩn hóa RMS và kẹp biên độ về $[-1.0, 1.0]$.
3. **Văn bản rỗng hoặc chỉ chứa khoảng trắng:**
   - Chuỗi vào `""` hoặc `"   \n\t"` ➔ Tự động `strip()`, trả về kết quả rỗng mà không gây crash tokenizer hay VITS vocoder.
4. **Từ ngữ H'Mông không có thanh điệu hoặc thanh điệu lạ:**
   - Từ không có phụ âm đuôi (thanh bằng / mid tone) ➔ Bộ chuẩn hóa hình thái RPA giữ nguyên từ gốc, không chèn ký tự rác.
   - Ký tự đặc biệt, biểu cảm emoji ➔ Lọc bỏ ký tự không nằm trong bảng chữ cái RPA.
5. **Đoạn âm thanh quá dài (> 15 giây):**
   - RingBuffer VAD tự động ngắt câu (endpointing) khi đạt ngưỡng thời gian tối đa để giải phóng bộ nhớ, tránh nghẽn RAM/VRAM.
6. **Mất kết nối WebSocket đột ngột:**
   - Client ngắt kết nối giữa chừng ➔ Server giải phóng buffer và task bất đồng bộ ngay lập tức, không để lại coroutine treo rò rỉ bộ nhớ.
7. **Dữ liệu upload sai định dạng hoặc hỏng:**
   - Tệp không phải audio hoặc JSON sai cấu trúc ➔ Bị chặn ngay từ tầng validate schema, trả về HTTP 400 Bad Request hoặc 422 Unprocessable Entity.

---

## 3. CHECKLIST NGHIỆM THU (ACCEPTANCE CRITERIA)

*Hệ thống chỉ được xem là hoàn thành khi đáp ứng đủ các tiêu chí dưới đây:*

- [ ] Toàn bộ mã nguồn chạy thành công không có lỗi cú pháp (`compileall` pass 100%).
- [ ] Module VietnameseSTT nhận dạng chính xác giọng nói từ file và microphone, hỗ trợ tiếng Việt có dấu.
- [ ] Module MT bảo toàn đúng quy tắc 8 thanh điệu RPA H'Mông (-b, -m, -d, -j, -v, -s, -g, mid).
- [ ] Module TTS tổng hợp ra âm thanh 16kHz rõ ràng, không bị méo tiếng, RTF < 0.1x.
- [ ] Độ trễ End-to-End P90 của toàn chuỗi S2S trên WebSocket nhỏ hơn 1.500 ms (1.5s).
- [ ] Tuân thủ tuyệt đối `RULES.md`: Không comment rác, type safety nghiêm ngặt, fail fast.
- [ ] Đặt file đúng vị trí quy định trong `ARCHITECTURE.md`.
- [ ] Toàn bộ 4 tầng test suite (`test_tier1` đến `test_tier4`, tổng cộng 63+ tests) pass 100%.
- [ ] Không sử dụng mock, stub, pass giả lập trong mã nguồn chính.
