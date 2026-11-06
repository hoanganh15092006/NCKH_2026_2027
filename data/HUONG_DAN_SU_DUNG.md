# HƯỚNG DẪN THÊM DỮ LIỆU TỪ ĐIỂN DỊCH (VIỆT - H'MÔNG)

Thư mục `data/` này dùng để bạn lưu trữ và bổ sung dữ liệu dịch cho hệ thống.
Hệ thống hỗ trợ tự động nhận diện và nạp dữ liệu từ **1 trong 3 định dạng** sau:

---

### 1. Thêm bằng file JSON: `data/user_dictionary.json`
Định dạng cặp `"tiếng việt": "tiếng h'mông rpa"`:
```json
{
  "tôi yêu bạn": "kuv hlub koj",
  "bạn đi đâu đấy": "koj mus qhov twg",
  "hẹn gặp lại": "sib ntsib dua"
}
```

---

### 2. Thêm bằng file Excel/CSV: `data/user_dictionary.csv`
Mỗi dòng là một cặp từ/câu, ngăn cách bởi dấu phẩy `,`:
```csv
tieng_viet,tieng_hmong
tôi yêu bạn,kuv hlub koj
bạn đi đâu đấy,koj mus qhov twg
hẹn gặp lại,sib ntsib dua
```

---

### 3. Thêm bằng file văn bản TXT: `data/user_dictionary.txt`
Mỗi dòng là một cặp câu, ngăn cách bởi phím **Tab** hoặc dấu phẩy `,`:
```txt
tôi yêu bạn	kuv hlub koj
bạn đi đâu đấy	koj mus qhov twg
hẹn gặp lại	sib ntsib dua
```

---

### Lưu ý quan trọng:
1. Bạn có thể dùng bất kỳ file nào trong 3 file trên. Khi hệ thống khởi động hoặc dịch thuật, nó sẽ tự động ưu tiên nạp từ điển của bạn trước.
2. Tiếng H'Mông nên viết theo chuẩn chữ cái La-tinh RPA (Romanized Popular Alphabet) có phụ âm chỉ thanh ở cuối từ (`-b`, `-j`, `-v`, `-s`, `-g`, `-m`, `-d`).
3. Nếu bạn muốn nạp ngay sau khi sửa file, chỉ cần báo cho AI hoặc khởi động lại server.
