import os
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ugocr import __version__
from ugocr.auth import verify_app_key
from ugocr.config import Settings
from ugocr.ocr.factory import PipelineFactory
from ugocr.schemas import OCRRecognizeResponse, TranslateTextRequest, TranslateTextResponse
from ugocr.translator import TextTranslator, normalize_lang_code


SUPPORTED_SOURCE_LANGS = {"uy", "ch", "kz"}


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or Settings.from_env()
    factory = PipelineFactory(active_settings)
    translator = TextTranslator(active_settings.vlm)
    app = FastAPI(title="UGOCR Offline OCR API", version=__version__)
    app.state.settings = active_settings
    app.state.factory = factory
    app.state.translator = translator

    @app.on_event("startup")
    async def _warmup_models() -> None:
        if active_settings.runtime.lazy_load_models:
            return
        include_table = os.getenv("UGOCR_WARMUP_TABLE", "true").strip().lower() in {"1", "true", "yes", "on"}
        factory.warmup(include_table=include_table)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(status_code=exc.status_code, content={"code": str(exc.status_code), "message": str(detail)})

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"code": "400", "message": "请求参数错误"})

    cors_origins_raw = os.getenv("UGOCR_CORS_ORIGINS", "*").strip()
    if cors_origins_raw == "*":
        cors_origins = ["*"]
    else:
        cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _extract_app_key(request: Request, appKey: str | None) -> str | None:
        if appKey:
            return appKey
        return request.headers.get("appKey") or request.headers.get("appkey") or request.headers.get("x-api-key")

    def _validate_app_key(request: Request, appKey: str | None) -> str:
        return verify_app_key(
            _extract_app_key(request, appKey),
            active_settings.auth.app_keys,
            active_settings.auth.enabled,
        )

    def _validate_image(file: UploadFile, source_lang: str) -> str:
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        filename = file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail={"code": "1001", "message": "图片格式不支持"},
            )
        try:
            normalized = normalize_lang_code(source_lang)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={"code": "400", "message": "无效的源语言代码"},
            )
        if normalized not in SUPPORTED_SOURCE_LANGS:
            raise HTTPException(
                status_code=400,
                detail={"code": "400", "message": "无效的源语言代码"},
            )
        return normalized

    def _normalize_translate_texts(payload: TranslateTextRequest) -> list[str]:
        if isinstance(payload.texts, list) and len(payload.texts) > 0:
            return [str(text) for text in payload.texts]
        if payload.text is not None and str(payload.text).strip():
            return [str(payload.text)]
        return []

    @app.post("/api/v1/ocr/recognize", response_model=OCRRecognizeResponse)
    async def ocr_recognize(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        source_lang: Annotated[str, Form(...)],
        appKey: Annotated[str | None, Header()] = None,
        use_vlm: Annotated[bool | None, Form()] = None,
        debug: Annotated[bool | None, Form()] = None,
    ):
        _validate_app_key(request, appKey)
        normalized_source_lang = _validate_image(file, source_lang)

        image_bytes = await file.read()
        max_bytes = active_settings.runtime.max_upload_mb * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail={"code": "1002", "message": "图片大小超限"},
            )
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail={"code": "1001", "message": "图片文件为空"},
            )

        try:
            result = factory.chinese.recognize(image_bytes, use_vlm=use_vlm)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "1003", "message": "识别失败"},
            ) from exc

        body = {
            "target_lang": normalized_source_lang,
            "result": [line.text for line in result.lines if line.text.strip()],
        }
        if debug is True:
            body.update(
                {
                    "use_vlm_requested": use_vlm is True,
                    "vlm_enabled": active_settings.vlm.enabled,
                    "corrected": result.corrected,
                    "warnings": result.warnings,
                }
            )
            return JSONResponse(content=body)
        return body

    @app.post("/api/v1/translate/text", response_model=TranslateTextResponse)
    async def translate_text(
        request: Request,
        payload: TranslateTextRequest,
        appKey: Annotated[str | None, Header()] = None,
    ):
        _validate_app_key(request, appKey)
        try:
            target_lang = normalize_lang_code(payload.target_lang)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={"code": "2002", "message": "不支持的语种"},
            )
        texts = _normalize_translate_texts(payload)
        if len(texts) == 0 or len(texts) > 20:
            raise HTTPException(
                status_code=400,
                detail={"code": "400", "message": "请求参数错误"},
            )
        if any(len(text) > 1000 for text in texts):
            raise HTTPException(
                status_code=400,
                detail={"code": "2003", "message": "文本长度超限"},
            )
        source_lang = None
        if payload.source_lang:
            try:
                source_lang = normalize_lang_code(payload.source_lang)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "2002", "message": "不支持的语种"},
                ) from None
        try:
            result = translator.translate(texts, target_lang=target_lang, source_lang=source_lang)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "2001", "message": "翻译服务异常"},
            ) from exc
        return {
            "source_lang": result.source_lang,
            "target_lang": result.target_lang,
            "texts": result.texts,
            "translate_results": result.translate_results,
        }

    @app.post("/api/v1/ocr/table")
    async def ocr_table(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        source_lang: Annotated[str, Form(...)],
        appKey: Annotated[str | None, Header()] = None,
        use_vlm: Annotated[bool | None, Form()] = None,
    ):
        _validate_app_key(request, appKey)
        _validate_image(file, source_lang)

        image_bytes = await file.read()
        max_bytes = active_settings.runtime.max_upload_mb * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail={"code": "1002", "message": "图片大小超限"},
            )
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail={"code": "1001", "message": "图片文件为空"},
            )

        try:
            xlsx_path, cells, corrected, warnings = factory.table.recognize_to_excel(image_bytes, use_vlm=use_vlm)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "1003", "message": "识别失败"},
            ) from exc

        return FileResponse(
            path=str(xlsx_path),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=xlsx_path.name,
        )

    return app


app = create_app()


def _resolve_bind_target(raw_host: str, raw_port: str | None) -> tuple[str, int]:
    host = raw_host.strip() or "0.0.0.0"
    port_text = (raw_port or "").strip()
    if "://" in host:
        parsed = urlparse(host)
        host = parsed.hostname or "0.0.0.0"
        if not port_text and parsed.port is not None:
            port_text = str(parsed.port)
    elif host.count(":") == 1:
        maybe_host, maybe_port = host.rsplit(":", 1)
        if maybe_port.isdigit():
            host = maybe_host or "0.0.0.0"
            if not port_text:
                port_text = maybe_port
    if host in {"*", "0"}:
        host = "0.0.0.0"
    return host, int(port_text or "8090")


def main() -> None:
    import uvicorn

    host, port = _resolve_bind_target(os.getenv("UGOCR_HOST", "0.0.0.0"), os.getenv("UGOCR_PORT"))
    log_level = os.getenv("UGOCR_LOG_LEVEL", "info").lower()
    certfile = os.getenv("UGOCR_SSL_CERTFILE", "").strip()
    keyfile = os.getenv("UGOCR_SSL_KEYFILE", "").strip()
    project_root = Path(__file__).resolve().parents[1]
    default_cert = project_root / "certs" / "server.crt"
    default_key = project_root / "certs" / "server.key"
    if not certfile and not keyfile and default_cert.exists() and default_key.exists():
        certfile = str(default_cert)
        keyfile = str(default_key)
    uvicorn.run(
        "ugocr.api:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
        ssl_certfile=certfile or None,
        ssl_keyfile=keyfile or None,
    )


if __name__ == "__main__":
    main()
