import test from 'node:test';
import assert from 'node:assert/strict';
import { formatBenchmarkRows } from '../src/benchmark.js';

const byLabel = rows => Object.fromEntries(rows.map(({ label, value }) => [label, value]));

test('formats measured benchmark values using the provided data', () => {
  const rows = byLabel(formatBenchmarkRows({
    model_variant: 'DA3-LARGE-1.1',
    device: 'cuda',
    video_duration_seconds: 56.86666666666667,
    source_fps: 30,
    source_frames: 1706,
    sampled_fps: 1,
    frames_extracted: 16,
    frames_processed: 16,
    frame_extraction_time_seconds: 2.3225585000000137,
    inference_time_seconds: 6.0157787000007374,
    export_time_seconds: 2.427173000000039,
    total_wall_clock_seconds: 43.447534599999926,
    frames_per_second: 0.3682602510661221,
    precision: 'bf16',
    num_chunks: 3,
    chunk_size: 8,
    chunk_overlap: 2,
    valid_depth_percentage: 100,
    mean_confidence: 1.7810730934143066,
    median_confidence: 1.3450732231140137,
    peak_gpu_memory_mb: 3879.18,
    glb_size_mb: 16.001272,
    ply_size_mb: 16.000244,
  }));

  assert.equal(rows.Model, 'DA3-LARGE-1.1');
  assert.equal(rows.Device, 'CUDA');
  assert.equal(rows['Video duration'], '56.87 s');
  assert.equal(rows.Source, '30 FPS');
  assert.equal(rows['Processed frames'], '16 / 16');
  assert.equal(rows['Frame processing'], '100%');
  assert.equal(rows['Throughput'], '0.368 FPS');
  assert.equal(rows['Peak GPU memory'], '3.88 GB');
  assert.equal(rows.GLB, '16.00 MB');
  assert.equal(rows.PLY, '16.00 MB');
});

test('omits unavailable, malformed, and non-finite metrics instead of showing zero', () => {
  const rows = byLabel(formatBenchmarkRows({
    status: 'completed',
    source_fps: 'Not measured',
    frames_processed: 0,
    frames_extracted: 0,
    valid_depth_percentage: null,
    mean_confidence: NaN,
    peak_gpu_memory_mb: 'not measured',
  }));

  assert.deepEqual(rows, {});
  assert.equal(formatBenchmarkRows(null).length, 0);
});
