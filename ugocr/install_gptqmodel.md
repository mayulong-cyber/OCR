# 安装 gptqmodel（7B AWQ 模型必需依赖）

## 方法1：安装 Visual Studio Build Tools（推荐）

1. 下载 Visual Studio Build Tools：
   https://aka.ms/vs

2. 安装时选择 "Desktop development with C++" 工作负载

3. 安装完成后重新打开终端，执行：
   ```cmd
   pip install gptqmodel
   ```

## 方法2：使用 conda 安装预编译版本

```cmd
conda install -c conda-forge gptqmodel
```

## 方法3：使用 pip 预编译包（如果有）

```cmd
pip install gptqmodel --only-binary :all:
```

## 安装成功后启动 7B 模型

```cmd
cd /d E:\OCR_Z_T\ugocr
set UGOCR_VLM_MODEL_NAME=qwen2.5-vl-7b-instruct-awq
set UGOCR_VLM_CPU_OFFLOAD=true
set UGOCR_VLM_MAX_GPU_LAYERS=4
set UGOCR_VLM_WARMUP=true
set UGOCR_VLM_MAX_CONCURRENCY=1
set UGOCR_VLM_MAX_NEW_TOKENS=256
set UGOCR_VLM_MAX_PIXELS=501760
python vlm_server.py
```

GPU 显存占用约 3.6GB（CPU offloading 后），RTX 4050 6GB 可以运行。
