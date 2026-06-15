from __future__ import annotations

from typing import Annotated

from fastapi import Header, HTTPException


def verify_app_key(
    app_key: str | None,
    valid_keys: list[str],
    auth_enabled: bool = True,
) -> str:
    if not auth_enabled:
        return "disabled"
    if not app_key:
        raise HTTPException(
            status_code=401,
            detail={"code": "401", "message": "未授权：缺少 appKey 请求头"},
        )
    if app_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail={"code": "401", "message": "未授权：appKey 无效"},
        )
    return app_key


def create_app_key_dependency(valid_keys: list[str], auth_enabled: bool = True):
    def _verify(appKey: Annotated[str | None, Header()] = None) -> str:
        return verify_app_key(appKey, valid_keys, auth_enabled)
    return _verify
