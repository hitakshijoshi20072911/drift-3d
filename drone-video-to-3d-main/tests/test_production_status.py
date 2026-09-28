import unittest

from sih_drone_pipeline.production_status import (
    RuntimeLedger,
    TerminalState,
    derive_terminal_state,
)


class TerminalStateTests(unittest.TestCase):
    def test_complete_requires_every_gate_and_deadline(self):
        state, reasons = derive_terminal_state(
            telemetry_valid=True,
            capture_sufficient=True,
            structural_artifacts_valid=True,
            substantive_geometry_valid=True,
            production_gates_pass=True,
            deadline_met=True,
        )
        self.assertEqual(state, TerminalState.COMPLETE)
        self.assertEqual(reasons, [])

    def test_degenerate_geometry_is_failed_not_partial(self):
        state, reasons = derive_terminal_state(
            telemetry_valid=True,
            capture_sufficient=True,
            structural_artifacts_valid=True,
            substantive_geometry_valid=False,
            production_gates_pass=False,
            deadline_met=True,
        )
        self.assertEqual(state, TerminalState.FAILED)
        self.assertIn("dense_geometry_is_degenerate", reasons)

    def test_deadline_stop_is_distinct_from_generic_partial(self):
        state, reasons = derive_terminal_state(
            telemetry_valid=True,
            capture_sufficient=True,
            structural_artifacts_valid=True,
            substantive_geometry_valid=True,
            production_gates_pass=False,
            deadline_met=False,
            stopped_for_deadline=True,
        )
        self.assertEqual(state, TerminalState.DEADLINE_LIMITED)
        self.assertIn("processing_stopped_at_deadline", reasons)


class RuntimeLedgerTests(unittest.TestCase):
    def test_experiments_do_not_contaminate_normal_pipeline_clock(self):
        ledger = RuntimeLedger()
        ledger.add_seconds("normal_pipeline", 600, "cells 7-9")
        ledger.add_seconds("experiment", 3600, "manual rescue trials")
        ledger.add_seconds("merge", 60, "component merge")
        summary = ledger.summary()
        self.assertEqual(summary["normal_pipeline_minutes"], 10)
        self.assertEqual(summary["experiment_minutes"], 60)
        self.assertEqual(summary["release_workflow_minutes"], 11)
        self.assertEqual(summary["session_accounted_minutes"], 71)

    def test_unknown_runtime_category_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeLedger().add_seconds("debug", 1, "not declared")


if __name__ == "__main__":
    unittest.main()

