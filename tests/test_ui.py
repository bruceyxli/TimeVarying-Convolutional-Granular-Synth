"""Streamlit smoke test: rendering and reruns must preserve the result."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class UITests(unittest.TestCase):
    def test_render_survives_parameter_and_download_reruns(self):
        ui = Path(__file__).resolve().parents[1] / "src" / "app" / "ui.py"
        app = AppTest.from_file(str(ui), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.slider(key="duration").set_value(2.0).run()
        next(button for button in app.button if button.label == "Render").click().run()
        self.assertEqual(len(app.exception), 0)
        result = app.session_state["last_render"]["wav"]
        self.assertGreater(len(result), 44)
        app.slider(key="wet").set_value(0.2).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(result, app.session_state["last_render"]["wav"])
        self.assertEqual(len(app.get("audio")), 1)
        self.assertEqual(len(app.get("download_button")), 1)

    def test_all_variants_and_preset_render(self):
        ui = Path(__file__).resolve().parents[1] / "src" / "app" / "ui.py"
        app = AppTest.from_file(str(ui), default_timeout=30).run()
        app.slider(key="duration").set_value(2.0).run()
        for label in ("Convolve → Granulate (A)", "Grains as IR (B)", "Standard"):
            next(w for w in app.selectbox if w.label == "Variant").select(label).run()
            next(w for w in app.button if w.label == "Render").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("last_render", app.session_state)
        next(w for w in app.selectbox if w.label == "Preset").select("Percussive microroom").run()
        next(w for w in app.button if w.label == "Apply preset").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["duration"], 2.0)


if __name__ == "__main__":
    unittest.main()
