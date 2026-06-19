# Chào mừng các bạn đến với Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

Repo này được chia thành ba phần rõ ràng:

- `src/`: bản scaffold dành cho sinh viên, chứa pseudocode và TODO để hoàn thiện
- `data/`: dữ liệu benchmark ở root để dùng cho cả benchmark chuẩn và stress benchmark

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Cách dùng repo này

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Sau đó làm việc trực tiếp với `src/` và `data/` ở root repo.

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.

---

## Kết quả benchmark và phân tích

Kết quả chạy thực tế với provider `openai / gpt-4o-mini` (offline-first, deterministic):

### Standard Benchmark (`data/conversations.json` — 10 hội thoại, ~10 turns mỗi hội thoại)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|:-----------------:|:-----------------------:|:--------------------:|:----------------:|:---------------------:|:-----------:|
| Baseline | 3,207             | 16,078                  | 0.000                | 0.071            | 0                     | 0           |
| Advanced | 4,272             | 24,279                  | **0.571**            | **0.643**        | 311                   | 0           |

### Long-Context Stress Benchmark (`data/advanced_long_context.json` — 1 hội thoại, 16 turns dày)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|:-----------------:|:-----------------------:|:--------------------:|:----------------:|:---------------------:|:-----------:|
| Baseline | 2,910             | 24,126                  | 0.000                | 0.100            | 0                     | 0           |
| Advanced | 3,196             | 11,525                  | **0.833**            | **0.867**        | 272                   | 4           |

---

### Phân tích kết quả

**1. Vì sao Advanced có recall tốt hơn Baseline?**

Baseline không có cơ chế lưu thông tin qua thread mới — mỗi `thread_id` mới là một trang giấy trắng. Advanced lưu facts (tên, nghề, nơi ở, style, v.v.) vào `User.md` sau mỗi lượt hội thoại. Khi recall question được hỏi ở một fresh thread, Advanced đọc lại `User.md` và trả lời đúng; Baseline không có gì để đọc nên trả lời sai hoàn toàn.

**2. Vì sao Advanced tốn token hơn Baseline ở hội thoại ngắn?**

Ở Standard Benchmark (thread ngắn ~10 turns), Advanced có `prompt tokens processed` cao hơn Baseline **+51%**. Nguyên nhân: mỗi lượt Advanced phải inject nội dung `User.md` + compact summary vào context — đây là overhead cố định dù hội thoại ngắn hay dài. Với thread ngắn, overhead này lớn hơn lợi ích của compact memory vì chưa đủ context để compact tiết kiệm được gì. Đây là trade-off bình thường của bất kỳ persistent memory system nào.

**3. Vì sao compact memory có lợi thế ở hội thoại dài?**

Ở Stress Benchmark (16 turns rất dài), Baseline phải kéo theo toàn bộ lịch sử hội thoại mỗi lượt — `prompt tokens processed` tăng tuyến tính theo độ dài thread (O(N²) tích lũy). Advanced dùng compact memory để thay thế phần cũ bằng một summary nhỏ cố định, giữ chỉ `keep_messages` messages gần nhất. Kết quả: Advanced xử lý ít context hơn Baseline **52%** ở hội thoại dài, trong khi recall vẫn đạt **0.833** nhờ User.md bảo tồn facts quan trọng.

Lưu ý quan trọng: compact memory tối ưu chủ yếu ở cột **`Prompt tokens processed`**, không phải `Agent tokens only`. Agent tokens (output) phụ thuộc vào độ dài câu trả lời, không liên quan trực tiếp đến memory architecture.

**4. File memory tăng trưởng ra sao và rủi ro đi kèm?**

Sau 10 hội thoại chuẩn, `User.md` của người dùng tăng lên **311 bytes**. Tốc độ tăng tỉ lệ với số facts mới được trích xuất mỗi session. Rủi ro chính:

- **Lưu sai fact**: nếu không có confidence threshold, agent có thể nhầm câu hỏi thành fact (“Mình tên gì?” → lưu “gì” vào `name`). Đã xử lý bằng cách bỏ qua message kết thúc `?`.
- **Conflict không được giải quyết**: khi người dùng sửa thông tin (“thực ra mình đang ở Đà Nẵng”), hệ thống phải detect correction trigger và override fact cũ thay vì giữ cả hai. Đã xử lý bằng `_CORRECTION_TRIGGERS`.
- **Nhiễu**: người dùng đùa hoặc nói giả định (“hay là chuyển sang product manager cho đỡ ngồi canh pipeline”) không phải fact thật. Đã lọc bằng `_NOISE_TRIGGERS`.
- **File phình to theo thời gian**: không có cơ chế cleanup hay size limit. Cần thêm memory decay hoặc periodic compaction cho production system.

---

### Kết luận câu chuyện memory

| Bước | Quan sát |
|------|---------|
| 1 | Baseline không nhớ dài hạn → recall = 0 qua thread mới |
| 2 | Advanced thêm `User.md` → recall tăng đáng kể |
| 3 | Hội thoại dài làm prompt cost của Baseline tăng mạnh (tuyến tính) |
| 4 | Compact memory kéo prompt cost của Advanced xuống ổn định |
| 5 | Hệ thống mạnh hơn nhưng phức tạp hơn — cần confidence threshold, conflict handling, và guardrail để không lưu sai |
