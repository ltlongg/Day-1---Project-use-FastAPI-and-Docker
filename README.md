Link Video: https://youtu.be/ga3uzlOPJlM
# 📝 Text Classification API

Sentiment Analysis API sử dụng **FastAPI** + **Hugging Face Transformers** + **MongoDB Compass** + **Docker**.

Model: **DistilBERT** fine-tuned trên SST-2 dataset (phân loại POSITIVE / NEGATIVE).
Kết quả phân loại được lưu vào **MongoDB** trên máy tính (host) để xem lại lịch sử.

## 🚀 Quick Start

### Chạy với Docker Compose (Khuyến nghị)

Ứng dụng Docker này được thiết lập để kết nối với **MongoDB Compass** (hoặc MongoDB server) đang chạy trực tiếp trên máy Windows của bạn qua port `27017`.
Hãy đảm bảo bạn đã mở MongoDB Compass và connect vào `mongodb://localhost:27017` trước khi chạy lệnh dưới.

```bash
# Build và chạy
docker-compose up --build

# Chạy ở background
docker-compose up --build -d

# Dừng
docker-compose down
```

### Chạy không dùng Docker (Trực tiếp bằng python)

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt

# Chạy server
uvicorn app.main:app --reload
```

## 📖 API Endpoints

### Health Check

```
GET /
GET /health
```

### Predict (lưu kết quả vào DB)

```
POST /predict
Content-Type: application/json

{
  "text": "I love this product! It's amazing!"
}
```

**Response:**

```json
{
  "text": "I love this product! It's amazing!",
  "prediction": {
    "label": "POSITIVE",
    "score": 0.9998
  },
  "all_scores": [
    {"label": "POSITIVE", "score": 0.9998},
    {"label": "NEGATIVE", "score": 0.0002}
  ]
}
```

### History (xem lịch sử phân loại)

```
GET /history?skip=0&limit=20
```

**Response:**

```json
[
  {
    "_id": "64f1a2b3c4d5e6f7a8b9c0d1",
    "text": "I love this product! It's amazing!",
    "label": "POSITIVE",
    "score": 0.9998,
    "created_at": "2026-03-04T22:00:00Z"
  }
]
```

## 📚 Swagger UI

Mở browser tại: **http://localhost:8888/docs**

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| ML Model | DistilBERT (Hugging Face) |
| Database | MongoDB (PyMongo) |
| Container | Docker + Docker Compose |
