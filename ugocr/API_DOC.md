# UGOCR API 接口文档

**版本**: 1.1.0  
**协议**: HTTP/HTTPS  
**数据格式**: `multipart/form-data`（OCR） / `application/json`（翻译）  
**字符编码**: UTF-8  
**超时建议**: 30 秒以内
**验证平台基础地址**: `https://172.25.144.4:8090`

---

## 1. OCR 识别接口

### 请求

```http
POST /api/v1/ocr/recognize
Content-Type: multipart/form-data
```

### 请求头

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| appKey | String | 是 | 认证凭证 |

兼容说明：服务同时兼容 `appKey`、`appkey`、`x-api-key` 三种请求头，规范写法仍为 `appKey`。

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | File | 是 | 图片文件流，支持格式：jpg、png、bmp |
| source_lang | String | 是 | 源语言代码：uy（维语）、ch（汉语）、kz（哈语） |
| use_vlm | Boolean | 否 | 是否启用 VLM 纠错，默认 false |

兼容说明：服务额外兼容 `zh -> ch`、`kk -> kz`，响应中统一返回规范语言码。

实现说明：当前部署默认验证场景为中文手写识别；`uy`、`kz` 语言码已在接口层兼容，但最终识别效果仍取决于已加载 OCR/VLM 模型能力。

### 成功响应

```json
{
  "target_lang": "ch",
  "result": [
    "第一行识别文字",
    "第二行识别文字"
  ]
}
```

### 响应字段

| 字段 | 类型 | 说明 |
|---|---|---|
| target_lang | String | 识别结果语言代码 |
| result | Array | 按行返回的识别结果 |

### 错误码

| 错误码 | 说明 |
|---|---|
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 1001 | 图片格式不支持 |
| 1002 | 图片大小超限（不超过 10MB） |
| 1003 | 识别失败 |
| 500 | 服务器内部错误 |

---

## 2. 智能文本翻译接口

### 请求

```http
POST /api/v1/translate/text
Content-Type: application/json
```

### 请求头

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| appKey | String | 是 | 认证凭证 |

兼容说明：服务同时兼容 `appKey`、`appkey`、`x-api-key` 三种请求头，规范写法仍为 `appKey`。

### 请求体

规范请求体：

```json
{
  "texts": [
    "想要翻译的文本 1",
    "想要翻译的文本 2"
  ],
  "target_lang": "ch"
}
```

兼容旧请求体：

```json
{
  "text": "单条文本",
  "source_lang": "uy",
  "target_lang": "ch"
}
```

### 请求字段

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| texts | Array | 是 | 待翻译文本数组，最大 20 条 |
| target_lang | String | 是 | 目标语言代码：uy、ch、kz |

兼容字段：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| text | String | 否 | 单条文本的兼容写法，会自动转成 `texts[0]` |
| source_lang | String | 否 | 源语言兼容写法；不传时由服务自动识别 |

兼容说明：服务额外兼容 `zh -> ch`、`kk -> kz`，响应中统一返回规范语言码。

### 成功响应

```json
{
  "source_lang": "uy",
  "target_lang": "ch",
  "texts": [
    "ﺋﯘﯾﻐﯘرﭼە"
  ],
  "translate_results": [
    "维语"
  ]
}
```

### 响应字段

| 字段 | 类型 | 说明 |
|---|---|---|
| source_lang | String | 识别出的源语言代码 |
| target_lang | String | 目标语言代码 |
| texts | Array | 原始文本数组 |
| translate_results | Array | 翻译结果数组，与 `texts` 一一对应 |

### 错误码

| 错误码 | 说明 |
|---|---|
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 2001 | 翻译服务异常 |
| 2002 | 不支持的语种 |
| 2003 | 文本长度超限（单条不超过 1000 字符） |
| 500 | 服务器内部错误 |

---

## 3. 表格识别接口

### 请求

```http
POST /api/v1/ocr/table
Content-Type: multipart/form-data
```

### 请求头

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| appKey | String | 是 | 认证凭证 |

兼容说明：服务同时兼容 `appKey`、`appkey`、`x-api-key` 三种请求头，规范写法仍为 `appKey`。

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | File | 是 | 图片文件流，支持格式：jpg、png、bmp |
| source_lang | String | 是 | 源语言代码：uy（维语）、ch（汉语）、kz（哈语） |
| use_vlm | Boolean | 否 | 是否启用 VLM 增强，默认 false |

### 成功响应

返回 Excel 文件（`.xlsx`），`Content-Type`：

```text
application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
```

### 错误码

| 错误码 | 说明 |
|---|---|
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 1001 | 图片格式不支持 |
| 1002 | 图片大小超限（不超过 10MB） |
| 1003 | 识别失败 |
| 500 | 服务器内部错误 |

---

## 4. 语言代码对照表

| 语言代码 | 语言名称 |
|---|---|
| uy | 维吾尔语 |
| ch | 汉语 |
| kz | 哈萨克语 |

---

## 5. 调用示例

### cURL

```bash
curl -k -X POST https://172.25.144.4:8090/api/v1/ocr/recognize \
  -H "appKey: test_key" \
  -F "file=@test.jpg" \
  -F "source_lang=ch" \
  -F "use_vlm=true"

curl -k -X POST https://172.25.144.4:8090/api/v1/translate/text \
  -H "appKey: test_key" \
  -H "Content-Type: application/json" \
  -d '{"texts":["ﺋﯘﯾﻐﯘرﭼە"],"target_lang":"ch"}'

curl -k -X POST https://172.25.144.4:8090/api/v1/ocr/table \
  -H "appKey: test_key" \
  -F "file=@table.jpg" \
  -F "source_lang=ch" \
  -F "use_vlm=true" \
  -o result.xlsx
```

### Windows CMD

```cmd
curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/ocr/recognize" -H "appKey: test_key" -F "file=@wtest3.png" -F "source_lang=ch" -F "use_vlm=true"

curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/translate/text" -H "appKey: test_key" -H "Content-Type: application/json" -d "{\"texts\":[\"سلام\"],\"target_lang\":\"ch\"}"

curl.exe -k -X POST "https://172.25.144.4:8090/api/v1/ocr/table" -H "appKey: test_key" -F "file=@table.png" -F "source_lang=ch" -F "use_vlm=true" -o result.xlsx
```
