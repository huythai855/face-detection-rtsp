# --- Stage 1: Build & Install dependencies ---
    FROM python:3.10-slim as builder

    WORKDIR /build
    
    # Cài đặt công cụ build cần thiết
    RUN apt-get update && apt-get install -y \
        build-essential \
        gcc \
        && rm -rf /var/lib/apt/lists/*
    
    COPY requirements.txt .
    # Install wheel để đóng gói gọn hơn, sau đó cài các gói vào thư mục /install
    RUN pip install --user --no-cache-dir -r requirements.txt
    
    # --- Stage 2: Final Runtime ---
    FROM python:3.10-slim
    
    WORKDIR /app
    
    # Chỉ cài các thư viện runtime cần thiết cho OpenCV (không cài compiler)
    RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
    # Copy các gói đã cài từ stage builder qua
    COPY --from=builder /root/.local /root/.local
    COPY . .
    
    # Đảm bảo script có thể tìm thấy các thư viện đã cài ở stage trước
    ENV PATH=/root/.local/bin:$PATH
    
    EXPOSE 8501
    
    # Chạy app
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]