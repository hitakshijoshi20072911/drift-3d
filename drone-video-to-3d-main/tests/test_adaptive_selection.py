import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None:
    from sih_drone_pipeline.adaptive_selection import (
        SelectionConfig, _features, verify_pair, choose_indices, extract_adaptive_keyframes,
    )
    from sih_drone_pipeline.telemetry import TelemetrySample


@unittest.skipIf(cv2 is None, 'OpenCV unavailable')
class AdaptiveSelectionTests(unittest.TestCase):
    def test_translated_texture_verified_but_unrelated_texture_rejected(self):
        rng = np.random.default_rng(20)
        texture = rng.integers(0, 256, (360, 680), dtype=np.uint8)
        unrelated = rng.integers(0, 256, (360, 640), dtype=np.uint8)
        orb = cv2.ORB_create(nfeatures=1200)
        a = _features(texture[:,:640], orb)
        b = _features(texture[:,20:660], orb)
        c = _features(unrelated, orb)
        self.assertTrue(verify_pair(a,b,SelectionConfig())['verified'])
        self.assertFalse(verify_pair(a,c,SelectionConfig())['verified'])

    def test_textureless_images_do_not_claim_connectivity(self):
        image = np.zeros((180,320),np.uint8)
        f = _features(image, cv2.ORB_create())
        self.assertFalse(verify_pair(f,f,SelectionConfig())['verified'])

    def test_overlap_loss_keeps_bridge_and_reports_actual_break(self):
        candidates = [{'time_s':i/10,'path_m':0,'sharpness':1} for i in range(6)]
        def pair(a,b):
            return {'verified':b-a<=2 and b != 5,'motion_fraction':0.01}
        selected, reasons = choose_indices(candidates,pair,SelectionConfig())
        self.assertIn(2,selected)
        self.assertIn('bridge_before_overlap_loss',reasons)
        self.assertIn('unresolved_visual_break',reasons)
        self.assertEqual(selected[-1],5)

    def test_motion_drives_more_frames_for_fast_scene(self):
        candidates = [{'time_s':i/3,'path_m':0,'sharpness':1} for i in range(90)]
        def select(speed):
            return choose_indices(candidates,lambda a,b:{'verified':True,
                'motion_fraction':speed*(b-a)},SelectionConfig())[0]
        self.assertGreater(len(select(.08)),len(select(.005)))

    def test_config_rejects_invalid_values(self):
        for kwargs in ({'candidate_fps':0},{'screening_timeout_s':float('nan')},
                       {'minimum_inlier_ratio':2}):
            with self.assertRaises(ValueError):
                SelectionConfig(**kwargs)

    def test_actual_video_decode_matches_source_frames_and_writes_graph(self):
        executable = shutil.which('ffmpeg')
        if executable is None:
            try:
                import imageio_ffmpeg
                executable = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                self.skipTest('FFmpeg unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Give the real packaged executable its usual command name.
            bindir = root/'bin'; bindir.mkdir()
            shutil.copy2(executable,bindir/('ffmpeg.exe' if os.name=='nt' else 'ffmpeg'))
            video = root/'moving.avi'
            writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(480,270))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(3)
            texture = rng.integers(0,256,(270,650,3),dtype=np.uint8)
            for index in range(120):
                writer.write(texture[:,index:index+480])
            writer.release()
            telemetry = [TelemetrySample(i,12,77+i*.000005,100) for i in range(13)]
            manifest = root/'frames.csv'
            with patch.dict(os.environ,{'PATH':str(bindir)+os.pathsep+os.environ['PATH']}):
                records = extract_adaptive_keyframes(video,root/'images',manifest_path=manifest,
                    telemetry_samples=telemetry,target_frames=10,max_width=480,
                    config=SelectionConfig(screening_timeout_s=45))
            report = json.loads(manifest.with_suffix('.selection.json').read_text())
            self.assertGreaterEqual(len(records),12)
            self.assertTrue(report['budget_conflict'])
            self.assertEqual(records[0].source_frame,0)
            self.assertEqual(records[-1].source_frame,119)
            self.assertEqual(report['verified_graph_components'],1)
            self.assertEqual(report['unverified_adjacent_links'],0)
            for row in records:
                self.assertAlmostEqual(row.time_s,row.source_frame/10,places=4)
                self.assertTrue((root/'images'/row.image_name).is_file())
            self.assertIsNone(report['visible_surface_coverage_fraction'])


if __name__ == '__main__':
    unittest.main()
