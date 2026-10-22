"""Bound request uploads before parsing and image decoding."""
import asyncio
from io import BytesIO
import tempfile
import warnings

from fastapi import HTTPException
from PIL import Image

MAX_BODY = 64 * 1024 * 1024
MAX_PIXELS = 89_478_485  # Pillow's standard decompression-bomb warning threshold.


class BoundedUploadMiddleware:
    def __init__(self, app, max_body=MAX_BODY, max_pending=4, timeout=120):
        self.app = app
        self.max_body = max_body
        self.max_pending = max_pending
        self.timeout = timeout
        self.pending = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)

        async def reject(status):
            body = b'{"detail":"Upload exceeds the service resource limit"}'
            await send({'type': 'http.response.start', 'status': status,
                        'headers': [(b'content-type', b'application/json'),
                                    (b'cache-control', b'no-store')]})
            await send({'type': 'http.response.body', 'body': body})

        for key, value in scope.get('headers', []):
            if key == b'content-length':
                try:
                    if int(value) > self.max_body:
                        return await reject(413)
                except ValueError:
                    return await reject(400)
        if self.pending >= self.max_pending:
            return await reject(429)
        self.pending += 1
        try:
            with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as buffered:
                total = 0

                async def collect():
                    nonlocal total
                    while True:
                        message = await receive()
                        if message['type'] == 'http.disconnect':
                            return 0
                        chunk = message.get('body', b'')
                        if total + len(chunk) > self.max_body:
                            return 413
                        buffered.write(chunk)
                        total += len(chunk)
                        if not message.get('more_body', False):
                            return 200

                try:
                    status = await asyncio.wait_for(collect(), self.timeout)
                except asyncio.TimeoutError:
                    return await reject(408)
                if status == 0:
                    return
                if status != 200:
                    return await reject(status)
                buffered.seek(0)
                remaining = total
                delivered = False

                async def replay():
                    nonlocal remaining, delivered
                    if delivered and not remaining:
                        return await receive()
                    chunk = buffered.read(min(64 * 1024, remaining))
                    remaining -= len(chunk)
                    delivered = True
                    return {'type': 'http.request', 'body': chunk, 'more_body': remaining > 0}

                await self.app(scope, replay, send)
        finally:
            self.pending -= 1


def validate_image_budget(images):
    """Inspect headers before templates allocate decoded frames."""
    total_pixels = 0
    for data in images:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as img:
                    frames = getattr(img, 'n_frames', 1)
                    total_pixels += img.width * img.height * frames
                    if frames > 1000 or total_pixels > MAX_PIXELS:
                        raise HTTPException(413, 'Decoded images exceed the pixel limit')
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            raise HTTPException(413, 'Decoded images exceed the pixel limit') from error
        except (OSError, ValueError) as error:
            raise HTTPException(400, 'Invalid image') from error
