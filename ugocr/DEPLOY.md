# UGOCR 生产环境部署指南

## 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU，显存 >= 8GB（推荐 RTX 3080/4070 或更高）
- **内存**: >= 16GB
- **硬盘**: >= 50GB（模型文件约 15GB）

### 软件要求
- **操作系统**: Linux（推荐 Ubuntu 20.04/22.04）或 Windows 10/11 Pro
- **Docker**: >= 20.10
- **Docker Compose**: >= 2.0
- **NVIDIA Container Toolkit**: 用于 GPU 支持

---

## 部署步骤

### 1. 安装 Docker

#### Linux (Ubuntu)
```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Windows
1. 下载安装 Docker Desktop: https://www.docker.com/products/docker-desktop/
2. 在 Docker Desktop 设置中启用 WSL2 后端
3. 确保 NVIDIA 驱动已安装

---

### 2. 准备模型文件

确保以下模型目录存在：
```
models/
├── ppocrv5_det/                    # 文字检测模型
├── ppocrv5_chinese_svtrv2_rec/     # 中文手写识别模型
├── ppocrv5_table_rec/              # 表格文字识别模型
├── slanet/                         # 表格结构识别模型
└── qwen2.5-vl-7b-instruct-awq/    # VLM 模型
```

---

### 3. 启动服务

```bash
# 进入项目目录
cd ugocr

# 启动所有服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f
```

Windows CMD 本地启动命令：
```cmd
powershell -ExecutionPolicy Bypass -File scripts\create_https_cert.ps1 -IpAddress 172.25.144.4
set UGOCR_CONFIG=configs\ugocr.prod.yaml && set UGOCR_USE_GPU=true && set UGOCR_DEVICE=cuda && set UGOCR_HOST=0.0.0.0 && set UGOCR_PORT=8090 && python -m ugocr
```

---

### 4. 验证服务

```bash
# 检查 VLM 服务
curl http://localhost:8001/health

# 测试 OCR 识别
curl -k -X POST https://172.25.144.4:8090/api/v1/ocr/recognize \
  -H "appKey: prod_key_2024" \
  -F "file=@test.jpg" \
  -F "source_lang=ch"

# 测试翻译接口
curl -k -X POST https://172.25.144.4:8090/api/v1/translate/text \
  -H "appKey: prod_key_2024" \
  -H "Content-Type: application/json" \
  -d '{"texts":["سلام"],"target_lang":"ch"}'

# 测试表格识别
curl -k -X POST https://172.25.144.4:8090/api/v1/ocr/table \
  -H "appKey: prod_key_2024" \
  -F "file=@table.jpg" \
  -F "source_lang=ch" \
  -o result.xlsx

# 5 秒响应时间回归测试
python scripts/smoke_api.py --config configs/ugocr.prod.yaml --only all --max-seconds 5
```

Windows CMD 验证命令：
```cmd
curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/ocr/recognize" -H "appKey: prod_key_2024" -F "file=@wtest2.png" -F "source_lang=ch"

curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/translate/text" -H "appKey: prod_key_2024" -H "Content-Type: application/json" -d "{\"texts\":[\"سلام\"],\"target_lang\":\"ch\"}"

curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/ocr/table" -H "appKey: prod_key_2024" -F "file=@table.png" -F "source_lang=ch" -o result.xlsx

python scripts\smoke_api.py --config configs\ugocr.prod.yaml --only all --max-seconds 5
```

---

## 常用命令

```bash
# 启动服务
docker compose up -d

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f ocr
docker compose logs -f vllm

# 更新服务
docker compose pull
docker compose up -d
```

---

## 配置说明

### 环境变量

在 `docker-compose.yml` 中可以配置以下环境变量：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `UGOCR_CONFIG` | 配置文件路径 | `configs/ugocr.prod.yaml` |
| `UGOCR_VLM_ENABLED` | 启用 VLM | `true` |
| `UGOCR_VLM_BASE_URL` | VLM 服务地址 | `http://vllm:8001/v1` |
| `UGOCR_VLM_MAX_TOKENS` | OCR 服务调用 VLM 的最大输出 token | `512` |
| `UGOCR_VLM_TIMEOUT_SEC` | OCR 服务等待 VLM 的超时时间（秒） | `25` |
| `UGOCR_VLM_WARMUP` | VLM 服务启动后执行小生成预热 | `true` |
| `UGOCR_VLM_MAX_CONCURRENCY` | VLM 服务最大并发请求数 | `1` |
| `UGOCR_VLM_MAX_NEW_TOKENS` | VLM 服务单次生成最大 token | `512` |
| `UGOCR_VLM_MAX_PIXELS` | VLM 服务接收图片的最大像素 token 约束 | `832*28*28` |
| `UGOCR_AUTH_ENABLED` | 启用认证 | `true` |
| `UGOCR_WARMUP_TABLE` | 启动时是否预热表格模型 | `true` |
| `UGOCR_PORT` | OCR 服务端口 | `8090` |
| `UGOCR_SSL_CERTFILE` | HTTPS 证书文件 | `certs/server.crt` |
| `UGOCR_SSL_KEYFILE` | HTTPS 私钥文件 | `certs/server.key` |

### 配置文件

生产环境配置文件：`configs/ugocr.prod.yaml`

说明：
- `lazy_load_models=false` 会在服务启动阶段预热 OCR 与表格模型，避免首个请求超时。
- 默认实时接口不走 VLM。只有显式传 `use_vlm=true` 才会调用 VLM。
- 翻译接口依赖当前配置的 VLM 服务；若 VLM 不可用，接口会返回 `2001 翻译服务异常`。

---

## 故障排除

### 1. GPU 内存不足
```bash
# 修改 docker-compose.yml 中的 gpu-memory-utilization
--gpu-memory-utilization 0.8  # 降低到 80%
```

### 2. VLM 服务启动失败
```bash
# 检查 GPU 是否可用
docker exec ugocr-vllm nvidia-smi

# 查看详细日志
docker compose logs vllm
```

### 3. OCR 服务连接 VLM 失败
```bash
# 检查网络
docker exec ugocr-ocr curl http://vllm:8001/health

# 检查配置
docker exec ugocr-ocr env | grep UGOCR
```

---

## 性能优化

### 1. 调整 VLM 并发
本地 `vlm_server.py` 默认单并发，适合 6GB 显存机器：
```cmd
set UGOCR_VLM_WARMUP=true && set UGOCR_VLM_MAX_CONCURRENCY=1 && set UGOCR_VLM_MAX_NEW_TOKENS=512 && python vlm_server.py
```

如果使用 vLLM，在 `docker-compose.yml` 中添加：
```yaml
command: >
  --model /models/qwen2.5-vl-7b-instruct-awq
  --port 8001
  --host 0.0.0.0
  --gpu-memory-utilization 0.9
  --max-model-len 4096
  --max-num-seqs 1  # 低显存优先稳定；显存充足再调高
```

### 2. 使用更小的模型
如果显存不足，可以使用 3B 模型：
```bash
# 下载 Qwen2.5-VL-3B-AWQ 模型
# 然后修改配置文件中的模型路径
```

### 3. 接口超时优化
```bash
# 保持默认 OCR-only 路径，不要在实时请求中传 use_vlm=true
# 生产环境使用启动预热
# python scripts/smoke_api.py --config configs/ugocr.prod.yaml --only all --max-seconds 5
# python scripts/warmup.py --api-url https://172.25.144.4:8090 --app-key test_key --chinese-image wtest2.png --use-vlm true --timeout 120
```

---

## 监控

### 健康检查
```bash
# VLM 服务
curl http://localhost:8001/health

# OCR/表格 5 秒回归
python scripts/smoke_api.py --config configs/ugocr.prod.yaml --only all --max-seconds 5

# Windows CMD: VLM 健康检查与 OCR+VLM 预热
curl.exe "http://127.0.0.1:8001/health"
python scripts\warmup.py --api-url https://172.25.144.4:8090 --app-key test_key --chinese-image wtest2.png --use-vlm true --timeout 120
```

### 日志查看
```bash
# 实时日志
docker compose logs -f

# 最近 100 行日志
docker compose logs --tail 100
```
