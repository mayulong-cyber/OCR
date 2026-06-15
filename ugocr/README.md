# UGOCR

离线 OCR API 服务，提供**手写汉字识别**、**智能文本翻译**和**表格图片识别**三类能力，支持 VLM 后处理纠错与翻译。

## 项目结构

```
ugocr/
├── ugocr/                          # 核心 Python 包
│   ├── __init__.py                 # 版本号
│   ├── __main__.py                 # 入口: python -m ugocr
│   ├── api.py                      # FastAPI 应用与全部 REST 端点
│   ├── auth.py                     # appKey 认证模块
│   ├── config.py                   # 配置模型 (YAML + 环境变量)
│   ├── schemas.py                  # Pydantic 请求/响应模型
│   ├── model_check.py              # 模型文件完整性校验
│   ├── dependencies.py             # 动态导入与 CUDA 兼容
│   └── ocr/
│       ├── factory.py              # 管线工厂 (惰性单例)
│       ├── chinese.py              # 手写汉字识别管线
│       ├── table.py                # 表格识别管线 + Excel 生成
│       ├── paddle_engine.py        # PaddleOCR/PaddleX 引擎封装
│       ├── postprocess.py          # VLM 后处理 (本地模型)
│       └── types.py                # 内部数据结构
│   └── utils/
│       ├── images.py               # 图片加载、裁剪、排序
│       ├── text.py                 # CJK 文本正则化
│       └── metrics.py              # Levenshtein 距离、字符精度
├── configs/
│   ├── ugocr.example.yaml          # 配置模板
│   └── ugocr.prod.yaml             # 生产环境配置
├── scripts/
│   ├── download_models.py          # 自动下载所需模型
│   ├── verify_models.py            # 校验模型文件完整性
│   └── smoke_api.py                # API 冒烟测试
├── vlm_server.py                   # 本地 VLM 服务
├── docker-compose.yml              # Docker Compose (GPU/CPU)
├── DEPLOY.md                       # 部署说明
├── API_DOC.md                      # 接口文档
└── README.md                       # 本文件
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/ocr/recognize` | 手写汉字 OCR 识别 |
| POST | `/api/v1/translate/text` | 智能文本翻译 |
| POST | `/api/v1/ocr/table` | 表格图片识别，返回 Excel 文件 |

详细的接口文档请参见 [API_DOC.md](API_DOC.md)

## 环境要求

- Python >= 3.10, < 3.13
- PaddlePaddle (CPU 或 GPU 版本)
- PaddleX >= 3.0
- 可选：Qwen2.5-VL-3B (用于 VLM 后处理纠错)

## 快速开始

### 1. 安装依赖

```bash
cd ugocr
pip install -e .
pip install -r requirements-gpu.txt  # 或 requirements-cpu.txt
pip install qwen-vl-utils bitsandbytes  # VLM 相关依赖
```

### 2. 下载模型

```bash
python scripts/download_models.py
```

或手动下载 VLM 模型：
```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/qwen2.5-vl-3b-instruct
```

### 3. 验证模型

```bash
python scripts/verify_models.py --config configs/ugocr.example.yaml
```

### 4. 启动服务

**生成 HTTPS 自签证书（首次运行一次）：**
```cmd
powershell -ExecutionPolicy Bypass -File scripts\create_https_cert.ps1 -IpAddress 172.25.144.4
```

**终端 1 - 启动 VLM 服务（可选）：**
```cmd
set UGOCR_VLM_WARMUP=true && set UGOCR_VLM_MAX_CONCURRENCY=1 && set UGOCR_VLM_MAX_NEW_TOKENS=512 && python vlm_server.py
```

**终端 2 - 启动 OCR 服务：**
```cmd
set UGOCR_CONFIG=configs\ugocr.example.yaml
python -m ugocr
```

**Windows CMD 推荐启动命令：**
```cmd
set UGOCR_CONFIG=configs\ugocr.example.yaml && set UGOCR_USE_GPU=true && set UGOCR_DEVICE=cuda && set UGOCR_HOST=0.0.0.0 && set UGOCR_PORT=8090 && python -m ugocr
```

服务启动后，`E:\接口验证\server.py` 会直接访问：

```text
https://172.25.144.4:8090/api/v1/ocr/recognize
```

当前验证范围只需要手写汉字识别和表格识别；表格验证平台入口已转发到 `/api/v1/ocr/table`。

排错说明：`UGOCR_HOST` 应写成绑定地址，例如 `0.0.0.0`，不要写完整 URL。代码已兼容误写成 `https://172.25.144.4:8090` 的情况，但推荐仍使用上面的启动命令。若验证平台报 `SSLEOFError`，通常表示 OCR 服务没有按 HTTPS 启动，请先确认 `certs/server.crt` 和 `certs/server.key` 存在。

### 5. 测试接口

```bash
# OCR 识别（不使用 VLM）
curl -k -X POST https://172.25.144.4:8090/api/v1/ocr/recognize \
  -H "appKey: test_key" \
  -F "file=@test.jpg" \
  -F "source_lang=ch" \
  -F "use_vlm=false"

# OCR 识别（使用 VLM 纠错）
curl -k -X POST https://172.25.144.4:8090/api/v1/ocr/recognize \
  -H "appKey: test_key" \
  -F "file=@test.jpg" \
  -F "source_lang=ch" \
  -F "use_vlm=true"

# 智能文本翻译
curl -k -X POST https://172.25.144.4:8090/api/v1/translate/text \
  -H "appKey: test_key" \
  -H "Content-Type: application/json" \
  -d '{"texts":["سلام"],"target_lang":"ch"}'

# 表格识别
curl -k -X POST https://172.25.144.4:8090/api/v1/ocr/table \
  -H "appKey: test_key" \
  -F "file=@table.jpg" \
  -F "source_lang=ch" \
  -o result.xlsx
```

**Windows CMD 测试命令：**
```cmd
curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/ocr/recognize" -H "appKey: test_key" -F "file=@wtest2.png" -F "source_lang=ch"

curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/ocr/recognize" -H "appKey: test_key" -F "file=@wtest2.png" -F "source_lang=ch" -F "use_vlm=true"

curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/translate/text" -H "appKey: test_key" -H "Content-Type: application/json" -d "{\"texts\":[\"سلام\"],\"target_lang\":\"ch\"}"

curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/ocr/table" -H "appKey: test_key" -F "file=@table.png" -F "source_lang=ch" -o result.xlsx

curl.exe "http://127.0.0.1:8001/health"
```

## 功能说明

### 手写汉字识别

```
POST /api/v1/ocr/recognize
```

原理：PP-OCRv5 检测 + Chinese SVTRv2 识别，可选 Qwen2.5-VL 后处理纠错。

说明：为满足接口响应时间要求，默认路径只走 OCR。只有显式传 `use_vlm=true` 时才调用 VLM。

兼容说明：鉴权头规范写法为 `appKey`，服务同时兼容 `appkey` 与 `x-api-key`。语言码规范写法为 `uy/ch/kz`，服务同时兼容旧写法 `zh`、`kk`，响应中统一返回规范语言码。

实现边界：当前默认落地并完成验证的是中文手写识别；`uy`、`kz` 的接口参数已兼容规范，但识别准确率仍取决于本地已部署的 OCR/VLM 模型。

请求头：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| appKey | String | 是 | 认证凭证 |

请求参数：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | File | 是 | 图片文件流，支持格式: jpg、png、bmp |
| source_lang | String | 是 | 源语言代码: uy（维语）、ch（汉语）、kz（哈语） |
| use_vlm | Boolean | 否 | 是否启用 VLM 纠错（默认 false；仅传 true 时启用） |

成功响应：
```json
{
    "target_lang": "ch",
    "result": ["第一行识别文字", "第二行识别文字"]
}
```

### 表格识别

```
POST /api/v1/ocr/table
```

原理：SLANet 表格结构识别 + Table Rec 单元格文字识别，可选 Qwen2.5-VL 增强。

请求参数同手写汉字识别。

成功响应：返回 Excel 文件（`.xlsx`）

### 智能文本翻译

```
POST /api/v1/translate/text
```

原理：通过当前配置的本地 VLM 服务执行离线文本翻译，自动识别源语言并返回规范语言码。

规范请求体：

```json
{
  "texts": ["سلام"],
  "target_lang": "ch"
}
```

兼容旧请求体：

```json
{
  "text": "سلام",
  "source_lang": "uy",
  "target_lang": "ch"
}
```

成功响应：

```json
{
  "source_lang": "uy",
  "target_lang": "ch",
  "texts": ["سلام"],
  "translate_results": ["你好"]
}
```

## VLM 后处理

支持可选的 Qwen2.5-VL 后处理，通过 `use_vlm` 参数控制：

- `use_vlm=true` — 启用 VLM 纠错/增强
- `use_vlm=false` 或不传 — 仅使用 PaddleOCR 原始结果，满足接口响应时间要求

说明：VLM 适合离线纠错或二阶段复核，不应作为规范接口的默认实时路径。VLM 服务启动时会默认执行一次小生成预热，并默认限制为单并发，避免首次请求和并发请求造成明显延迟或显存峰值。

### VLM 模型

| 模型 | 大小 | 说明 |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | ~7.5GB | 推荐，适合 6GB 显存 |
| Qwen2.5-VL-7B-Instruct-AWQ | ~7GB | 需要 gptqmodel，显存要求更高 |

## 接口文档

详细的接口文档请参见 [API_DOC.md](API_DOC.md)

## 配置文件

所有配置通过 YAML 文件管理，环境变量会覆盖 YAML 值。

### 配置示例

```yaml
runtime:
  use_gpu: true
  device: cuda
  lazy_load_models: false
  upload_dir: data/uploads
  output_dir: data/outputs
  max_upload_mb: 10

paddle:
  det_model_dir: models/ppocrv5_det
  chinese_rec_model_dir: models/ppocrv5_chinese_svtrv2_rec
  table_rec_model_dir: models/ppocrv5_table_rec
  table_model_dir: models/slanet
  use_angle_cls: true
  det_limit_side_len: 1920

vlm:
  enabled: true
  provider: openai
  base_url: http://localhost:8001/v1
  model: qwen2.5-vl-3b-instruct
  api_key: local-offline-key
  max_tokens: 512
  temperature: 0.0
  request_timeout_sec: 120

auth:
  enabled: true
  app_keys:
    - "test_key"
    - "prod_key_2024"
```

说明：`lazy_load_models=false` 会在服务启动阶段预热 OCR 和表格模型，把模型加载时间前置，避免首个请求超时。

### 环境变量一览

| 变量 | 说明 | 默认值 |
|---|---|---|
| `UGOCR_CONFIG` | YAML 配置文件路径 | `configs/ugocr.example.yaml` |
| `UGOCR_USE_GPU` | 使用 GPU | false |
| `UGOCR_DEVICE` | 设备类型 | cpu |
| `UGOCR_LAZY_LOAD_MODELS` | 惰性加载模型；生产建议 false 以启动预热模型 | false |
| `UGOCR_MAX_UPLOAD_MB` | 最大上传文件 (MB) | 10 |
| `UGOCR_AUTH_ENABLED` | 启用认证 | true |
| `UGOCR_VLM_ENABLED` | 启用 VLM | false |
| `UGOCR_VLM_BASE_URL` | VLM API 地址 | `http://localhost:8001/v1` |
| `UGOCR_VLM_MAX_TOKENS` | OCR 服务调用 VLM 的最大输出 token | 512 |
| `UGOCR_VLM_TIMEOUT_SEC` | OCR 服务等待 VLM 的超时时间（秒） | 25 |
| `UGOCR_VLM_WARMUP` | VLM 服务启动后执行小生成预热 | true |
| `UGOCR_VLM_MAX_CONCURRENCY` | VLM 服务最大并发请求数 | 1 |
| `UGOCR_VLM_MAX_NEW_TOKENS` | VLM 服务单次生成最大 token | 512 |
| `UGOCR_VLM_MAX_PIXELS` | VLM 服务接收图片的最大像素 token 约束 | `832*28*28` |
| `UGOCR_HOST` | API 绑定地址 | `0.0.0.0` |
| `UGOCR_PORT` | API 端口 | `8090` |
| `UGOCR_SSL_CERTFILE` | HTTPS 证书文件 | `certs/server.crt`（存在时自动使用） |
| `UGOCR_SSL_KEYFILE` | HTTPS 私钥文件 | `certs/server.key`（存在时自动使用） |
| `UGOCR_LOG_LEVEL` | 日志级别 | info |
| `UGOCR_CORS_ORIGINS` | CORS 允许源 | `*` |

## 模型目录

| 模型 | 目录 | 用途 |
|---|---|---|
| PP-OCRv5 Det | `models/ppocrv5_det/` | 文字检测 (共用) |
| Chinese SVTRv2 Rec | `models/ppocrv5_chinese_svtrv2_rec/` | 中文手写识别 |
| Table Rec | `models/ppocrv5_table_rec/` | 表格文字识别 |
| SLANet | `models/slanet/` | 表格结构识别 |
| Qwen2.5-VL-3B | `models/qwen2.5-vl-3b-instruct/` | VLM 后处理 (可选) |

## 运维脚本

```bash
# 下载全部所需模型
python scripts/download_models.py

# 校验模型文件完整性
python scripts/verify_models.py --config configs/ugocr.example.yaml

# 5 秒响应时间冒烟测试
python scripts/smoke_api.py --config configs/ugocr.example.yaml --only all --max-seconds 5

# 翻译接口单独回归（需先启动 VLM）
python scripts/smoke_api.py --config configs/ugocr.example.yaml --only translate --max-seconds 30

# 真实接口预热，可选择是否调用 VLM
python scripts/warmup.py --api-url https://172.25.144.4:8090 --app-key test_key --chinese-image wtest2.png --table-image table.png --use-vlm false
```

**Windows CMD 回归命令：**
```cmd
python scripts\verify_models.py --config configs\ugocr.example.yaml
python scripts\smoke_api.py --config configs\ugocr.example.yaml --only all --max-seconds 5
python scripts\smoke_api.py --config configs\ugocr.example.yaml --only translate --max-seconds 30
python scripts\warmup.py --api-url https://172.25.144.4:8090 --app-key test_key --chinese-image wtest2.png --use-vlm true --timeout 120
```

## 生产环境部署

详细的部署说明请参见 [DEPLOY.md](DEPLOY.md)
