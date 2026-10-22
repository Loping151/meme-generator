import asyncio
import importlib
import io
import os
import struct
import unittest
import zlib
from unittest.mock import patch

from PIL import Image
from fastapi import HTTPException

module = importlib.import_module(os.environ.get('INPUT_GUARD_MODULE', 'meme_generator.public_input'))


class UploadTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, chunks, *, limit=64, headers=(), timeout=1, receive_override=None):
        sent, received = [], []
        messages = iter([{'type': 'http.request', 'body': chunk, 'more_body': i < len(chunks)-1}
                         for i, chunk in enumerate(chunks)])

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        async def application(scope, receive, send):
            while True:
                message = await receive()
                received.append(message['body'])
                if not message.get('more_body'):
                    break
            await send({'type': 'http.response.start', 'status': 200})
            await send({'type': 'http.response.body', 'body': b'ok'})

        guard = module.BoundedUploadMiddleware(application, max_body=limit, timeout=timeout)
        await guard({'type':'http','method':'POST','headers':headers}, receive_override or receive, send)
        self.assertEqual(guard.pending, 0)
        return sent, received

    async def test_chunked_upload_rejected_before_parser(self):
        sent, received = await self.request([b'a'*40, b'b'*25])
        self.assertEqual(sent[0]['status'], 413)
        self.assertEqual(received, [])

    async def test_declared_oversize_never_read(self):
        async def receive():
            self.fail('oversized body must not be read')
        sent, received = await self.request([], headers=[(b'content-length',b'65')], receive_override=receive)
        self.assertEqual(sent[0]['status'], 413)

    async def test_spooled_payload_replayed_exactly_in_bounded_chunks(self):
        data = b'abcd' * 300_000
        sent, chunks = await self.request([data[:400000],data[400000:]], limit=len(data))
        self.assertEqual(sent[0]['status'],200)
        self.assertEqual(b''.join(chunks), data)
        self.assertLessEqual(max(map(len,chunks)),64*1024)

    async def test_slow_upload_times_out_and_releases_slot(self):
        async def receive():
            await asyncio.sleep(1)
        sent, received = await self.request([],timeout=0.01,receive_override=receive)
        self.assertEqual(sent[0]['status'],408)
        self.assertEqual(received,[])

    async def test_disconnect_releases_slot_without_response(self):
        async def receive():
            return {'type':'http.disconnect'}
        sent, received = await self.request([],receive_override=receive)
        self.assertEqual(sent,[])
        self.assertEqual(received,[])

    async def test_admission_is_bounded_before_body_read(self):
        gate=asyncio.Event(); sent=[]
        async def receive():
            await gate.wait()
            return {'type':'http.request','body':b'','more_body':False}
        async def send(message):sent.append(message)
        async def app(scope,receive,send):await receive()
        guard=module.BoundedUploadMiddleware(app,max_pending=2)
        scope={'type':'http','method':'POST'}
        tasks=[asyncio.create_task(guard(scope,receive,send)) for _ in range(2)]
        await asyncio.sleep(0.01)
        await guard(scope,receive,send)
        self.assertEqual(sent[0]['status'],429)
        gate.set();await asyncio.gather(*tasks)
        self.assertEqual(guard.pending,0)

    async def test_non_upload_passes_through(self):
        called=[]
        async def app(scope,receive,send):called.append(scope['type'])
        guard=module.BoundedUploadMiddleware(app)
        await guard({'type':'websocket'},None,None)
        await guard({'type':'http','method':'GET'},None,None)
        self.assertEqual(called,['websocket','http'])


class PixelTests(unittest.TestCase):
    def png(self):
        out=io.BytesIO();Image.new('RGB',(32,32),'blue').save(out,'PNG');return out.getvalue()

    def test_ordinary_images_pass(self):
        module.validate_image_budget([self.png()])
        module.validate_image_budget([])

    def test_aggregate_image_budget(self):
        with patch.object(module,'MAX_PIXELS',1500),self.assertRaises(HTTPException) as caught:
            module.validate_image_budget([self.png(),self.png()])
        self.assertEqual(caught.exception.status_code,413)

    def test_animated_frames_are_counted(self):
        frames=[Image.new('RGB',(16,16),color) for color in ['red','green','blue']]
        out=io.BytesIO();frames[0].save(out,'GIF',save_all=True,append_images=frames[1:])
        with patch.object(module,'MAX_PIXELS',500),self.assertRaises(HTTPException) as caught:
            module.validate_image_budget([out.getvalue()])
        self.assertEqual(caught.exception.status_code,413)

    def test_large_header_rejected_without_pixel_allocation(self):
        def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
        data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',10000,10000,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''))+chunk(b'IEND',b'')
        with self.assertRaises(HTTPException) as caught:module.validate_image_budget([data])
        self.assertEqual(caught.exception.status_code,413)

    def test_invalid_image_rejected(self):
        with self.assertRaises(HTTPException) as caught:module.validate_image_budget([b'bad'])
        self.assertEqual(caught.exception.status_code,400)


if __name__=='__main__':unittest.main()
