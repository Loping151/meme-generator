import json
from datetime import datetime
from typing import Any, Literal, Optional

import filetype
from fastapi import Depends, FastAPI, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, ValidationError

from meme_generator.compat import model_dump, model_json_schema, type_validate_python
from meme_generator.config import meme_config
from meme_generator.exception import (
    ArgModelMismatch,
    MemeGeneratorException,
    NoSuchMeme,
)
from meme_generator.log import LOGGING_CONFIG, setup_logger
from meme_generator.manager import get_meme, get_meme_keys, get_memes
from meme_generator.meme import CommandShortcut, Meme, MemeArgsModel, ParserOption
from meme_generator.utils import MemeProperties, render_meme_list, run_sync
from meme_generator.version import __version__

from meme_generator.public_input import BoundedUploadMiddleware, validate_image_budget

app = FastAPI()
app.add_middleware(BoundedUploadMiddleware)

# 允许浏览器跨站调用: 本服务本来就是公开的(任何客户端 curl 即可), CORS 只影响浏览器,
# 加上它不减少任何保护, 只是让 心华.com 的表情工坊页面可以直连这里出图 ——
# 否则那些图要绕经 心华.com 的服务器中转一趟, 白占一份带宽。
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# 允许跨站的站点页面(心华.com / 维里奈.com 及备用入口)。下面的公网限频对这些 Origin 发来的请求放宽一档
PUBLIC_ORIGINS = [
    "https://xn--xkrt30a.com",
    "https://www.xn--xkrt30a.com",
    "https://xn--ntsp09fr3m.com",  # 维里奈.com（小维官网，同人帖「小维表情包制作」直连出图）
    "https://www.xn--ntsp09fr3m.com",
    "https://bot.loping151.site",
    "https://loping151.site",  # 备用入口 loping151.site/bot
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=PUBLIC_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 公网限频: 只管从 frp 进来的生成请求(frpc 的 https2http 插件给每个请求加了 x-from-where: frp,
# 并把真实客户端地址写进 X-Forwarded-For), 本机 / bot 直连 127.0.0.1 不受影响。
# 生成一张约一秒、都在一块卡上算, 额度只能给到这个量级; 预览 / info / keys 不计。
from collections import defaultdict, deque  # noqa: E402
import time as _time  # noqa: E402
from fastapi import Request as _Request  # noqa: E402
from fastapi.responses import JSONResponse as _JSONResponse  # noqa: E402

# 两档: 自家站点页面(Origin 在白名单里)连着改字也是一次一张, 给宽一点; 其它(curl / 机器人)按 API 的量给。
# Origin 谁都能伪造, 这只是给自家页面让路, 不是安全措施。
PUBLIC_RENDER_PER_MINUTE = 30
PUBLIC_RENDER_PER_DAY = 600
SITE_RENDER_PER_MINUTE = 60
SITE_RENDER_PER_DAY = 2000
_pub_minute: dict[str, deque] = defaultdict(deque)
_pub_day: dict[str, deque] = defaultdict(deque)


def _public_allow(ip: str, per_minute: int, per_day: int) -> bool:
    now = _time.time()
    m, d = _pub_minute[ip], _pub_day[ip]
    while m and now - m[0] >= 60:
        m.popleft()
    while d and now - d[0] >= 86400:
        d.popleft()
    if len(m) >= per_minute or len(d) >= per_day:
        return False
    m.append(now)
    d.append(now)
    if len(_pub_minute) > 4096:
        for k in [k for k, v in _pub_minute.items() if not v]:
            del _pub_minute[k]
        for k in [k for k, v in _pub_day.items() if not v]:
            del _pub_day[k]
    return True


@app.middleware("http")
async def _public_rate_limit(request: _Request, call_next):
    if (
        request.method == "POST"
        and request.url.path.startswith("/memes/")
        and request.headers.get("x-from-where") == "frp"
    ):
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or (request.client.host if request.client else "?"))
        origin = request.headers.get("origin", "")
        site = origin in PUBLIC_ORIGINS
        per_minute, per_day = (SITE_RENDER_PER_MINUTE, SITE_RENDER_PER_DAY) if site else (PUBLIC_RENDER_PER_MINUTE, PUBLIC_RENDER_PER_DAY)
        if not _public_allow(("site:" if site else "api:") + ip, per_minute, per_day):
            return _JSONResponse(
                {"detail": "生成请求过于频繁", "hint": f"每分钟 {per_minute} 次, 每天 {per_day} 次; 预览不计入。"},
                status_code=429,
                headers={"Retry-After": "60", "Access-Control-Allow-Origin": origin or "*"},
            )
    return await call_next(request)


class MemeArgsResponse(BaseModel):
    args_model: dict[str, Any]
    args_examples: list[dict[str, Any]]
    parser_options: list[ParserOption]


class MemeParamsResponse(BaseModel):
    min_images: int
    max_images: int
    min_texts: int
    max_texts: int
    default_texts: list[str]
    args_type: Optional[MemeArgsResponse] = None


class MemeInfoResponse(BaseModel):
    key: str
    params_type: MemeParamsResponse
    keywords: list[str]
    shortcuts: list[CommandShortcut]
    tags: set[str]
    date_created: datetime
    date_modified: datetime


def register_router(meme: Meme):
    if args_type := meme.params_type.args_type:
        args_model = args_type.args_model
    else:
        args_model = MemeArgsModel

    def args_checker(
        args: Optional[str] = Form(default=json.dumps(model_dump(args_model()))),
    ):
        if not args:
            return MemeArgsModel()
        try:
            model = type_validate_python(args_model, json.loads(args))
        except ValidationError as e:
            e = ArgModelMismatch(str(e))
            raise HTTPException(status_code=552, detail=e.message)
        return model

    @app.post(f"/memes/{meme.key}/")
    async def _(
        images: list[UploadFile] = [],
        texts: list[str] = meme.params_type.default_texts,
        args: args_model = Depends(args_checker),  # type: ignore
    ):
        imgs: list[bytes] = []
        for image in images:
            imgs.append(await image.read())
        validate_image_budget(imgs)

        texts = [text for text in texts if text]

        assert isinstance(args, args_model)

        try:
            result = await run_sync(meme)(
                images=imgs, texts=texts, args=model_dump(args)
            )
        except MemeGeneratorException as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

        content = result.getvalue()
        media_type = str(filetype.guess_mime(content)) or "text/plain"
        return Response(content=content, media_type=media_type)


class MemeKeyWithProperties(BaseModel):
    meme_key: str
    disabled: bool = False
    labels: list[Literal["new", "hot"]] = []


default_meme_list = [
    MemeKeyWithProperties(meme_key=meme.key)
    for meme in sorted(get_memes(), key=lambda meme: meme.key)
]


class RenderMemeListRequest(BaseModel):
    meme_list: list[MemeKeyWithProperties] = default_meme_list
    text_template: str = "{keywords}"
    add_category_icon: bool = True


def register_routers():
    @app.post("/memes/render_list")
    def _(params: RenderMemeListRequest = RenderMemeListRequest()):
        try:
            meme_list = [
                (
                    get_meme(p.meme_key),
                    MemeProperties(disabled=p.disabled, labels=p.labels),
                )
                for p in params.meme_list
            ]
        except NoSuchMeme as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

        result = render_meme_list(
            meme_list,
            text_template=params.text_template,
            add_category_icon=params.add_category_icon,
        )
        content = result.getvalue()
        media_type = str(filetype.guess_mime(content)) or "text/plain"
        return Response(content=content, media_type=media_type)

    @app.get("/meme/version")
    def _():
        return __version__

    @app.get("/memes/keys")
    def _():
        return get_meme_keys()

    @app.get("/memes/{key}/info")
    def _(key: str):
        try:
            meme = get_meme(key)
        except NoSuchMeme as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

        args_type_response = None
        if args_type := meme.params_type.args_type:
            args_model = args_type.args_model
            args_type_response = MemeArgsResponse(
                args_model=model_json_schema(args_model),
                args_examples=[
                    model_dump(example) for example in args_type.args_examples
                ],
                parser_options=args_type.parser_options,
            )

        return MemeInfoResponse(
            key=meme.key,
            params_type=MemeParamsResponse(
                min_images=meme.params_type.min_images,
                max_images=meme.params_type.max_images,
                min_texts=meme.params_type.min_texts,
                max_texts=meme.params_type.max_texts,
                default_texts=meme.params_type.default_texts,
                args_type=args_type_response,
            ),
            keywords=meme.keywords,
            shortcuts=meme.shortcuts,
            tags=meme.tags,
            date_created=meme.date_created,
            date_modified=meme.date_modified,
        )

    @app.get("/memes/{key}/preview")
    async def _(key: str):
        try:
            meme = get_meme(key)
            result = await run_sync(meme.generate_preview)()
        except MemeGeneratorException as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

        content = result.getvalue()
        media_type = str(filetype.guess_mime(content)) or "text/plain"
        return Response(content=content, media_type=media_type)

    for meme in sorted(get_memes(), key=lambda meme: meme.key):
        register_router(meme)


def run_server():
    import uvicorn

    register_routers()
    uvicorn.run(
        app,
        host=meme_config.server.host,
        port=meme_config.server.port,
        log_config=LOGGING_CONFIG,
    )


if __name__ == "__main__":
    setup_logger()
    run_server()
