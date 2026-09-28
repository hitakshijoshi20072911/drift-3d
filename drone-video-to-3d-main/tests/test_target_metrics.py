import unittest
from sih_drone_pipeline.target_metrics import duration_deadline_minutes, requirement_report


class TargetTests(unittest.TestCase):
    def test_duration_scaling_and_override(self):
        self.assertEqual(duration_deadline_minutes(600),15)
        self.assertEqual(duration_deadline_minutes(1200),30)
        self.assertEqual(duration_deadline_minutes(120),3.5)
        self.assertEqual(duration_deadline_minutes(1200,15),15)
        with self.assertRaises(ValueError):
            duration_deadline_minutes(0)

    def test_missing_accuracy_and_surface_coverage_never_pass(self):
        report = requirement_report(duration_s=600,elapsed_s=800,deadline_minutes=15,
            selection={'selected_frames':200,'candidate_visual_support_fraction':1.0},
            component_reports=[{'status':'reconstructed'}],artifacts={},merge_pass=True)
        self.assertTrue(report['checks']['runtime'])
        self.assertIsNone(report['checks']['spatial_accuracy'])
        self.assertIsNone(report['checks']['entire_visible_scene'])
        self.assertFalse(report['production_ready'])

    def test_runtime_is_strict_and_checkpoint_evidence_is_separate(self):
        report = requirement_report(duration_s=600,elapsed_s=900,deadline_minutes=15,
            selection={},component_reports=[],artifacts={},
            checkpoints={'rmse_3d_m':0.8,'checkpoint_count':10})
        self.assertFalse(report['checks']['runtime'])
        self.assertTrue(report['checks']['spatial_accuracy'])
        self.assertFalse(report['production_ready'])


if __name__ == '__main__':
    unittest.main()
