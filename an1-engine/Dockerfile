FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3","bench/zipf_reuse_benchmark.py","--U","8192","--out","2048","--kernels","64","--steps","600","--batch","8192","--alpha","1.6","--fp16","--sort-gather","--graph-gather"]
