import io
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
import numpy as np
import soundfile as sf
from src.app.web import AudioServer, demo_source
from src.app.audio_io import decode_audio


class SourcePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = AudioServer(("127.0.0.1", 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def read(self, suffix):
        with urlopen(self.url + suffix, timeout=5) as response:
            self.assertEqual(response.headers["Content-Type"], "audio/wav")
            return sf.read(io.BytesIO(response.read()), dtype="float32")

    def test_demo_preview_contains_the_actual_renderer_source(self):
        data, rate = self.read("/api/source-audio/demo")
        self.assertEqual(rate, 48000)
        np.testing.assert_array_equal(data, demo_source(48000))

    def test_imported_preview_uses_the_renderer_decode_path(self):
        samples = np.column_stack((np.linspace(-.3,.3,4410), np.linspace(.1,-.1,4410)))
        buffer = io.BytesIO()
        sf.write(buffer, samples, 44100, format="WAV", subtype="FLOAT")
        payload = buffer.getvalue()
        token = self.server.store(self.server.sources, payload)
        data, rate = self.read("/api/source-audio/" + token)
        self.assertEqual(rate, 48000)
        np.testing.assert_array_equal(data, decode_audio(payload, 48000))

    def test_expired_source_is_an_error_not_demo_substitution(self):
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.url + "/api/source-audio/missing", timeout=5)
        self.assertEqual(raised.exception.code, 404)
