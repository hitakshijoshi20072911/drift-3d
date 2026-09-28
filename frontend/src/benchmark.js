const numeric = value => typeof value === 'number' && Number.isFinite(value);
const text = value => typeof value === 'string' && value.trim() && !/^not measured$/i.test(value.trim());

function fixed(value, digits = 2) {
  if (!numeric(value)) return null;
  return value.toFixed(digits);
}

function compact(value, digits = 2) {
  const rendered = fixed(value, digits);
  return rendered === null ? null : rendered.replace(/\.?0+$/, '');
}

/** Convert an optional API benchmark object into display-ready sidebar rows. */
export function formatBenchmarkRows(benchmark) {
  if (!benchmark || typeof benchmark !== 'object' || Array.isArray(benchmark)) return [];
  const rows = [];
  const add = (label, value) => {
    if (value !== null && value !== undefined && value !== '') rows.push({ label, value: String(value) });
  };
  const addNumber = (label, value, digits = 2, suffix = '') => {
    const rendered = fixed(value, digits);
    if (rendered !== null) add(label, `${rendered}${suffix}`);
  };

  if (text(benchmark.model_variant)) add('Model', benchmark.model_variant);
  if (text(benchmark.device)) add('Device', benchmark.device.toUpperCase());
  addNumber('Video duration', benchmark.video_duration_seconds, 2, ' s');
  if (numeric(benchmark.source_fps)) add('Source', `${compact(benchmark.source_fps)} FPS`);
  if (numeric(benchmark.source_frames)) add('Source frames', benchmark.source_frames);
  addNumber('Sampled FPS', benchmark.sampled_fps, 2);

  if (numeric(benchmark.frames_processed) && numeric(benchmark.frames_extracted) && benchmark.frames_extracted > 0) {
    add('Processed frames', `${benchmark.frames_processed} / ${benchmark.frames_extracted}`);
    if (benchmark.frames_extracted > 0) {
      addNumber('Frame processing', 100 * benchmark.frames_processed / benchmark.frames_extracted, 0, '%');
    }
  }
  addNumber('Frame extraction', benchmark.frame_extraction_time_seconds, 2, ' s');
  addNumber('Inference', benchmark.inference_time_seconds, 2, ' s');
  addNumber('Export', benchmark.export_time_seconds, 2, ' s');
  addNumber('Total processing', benchmark.total_wall_clock_seconds, 2, ' s');
  addNumber('Throughput', benchmark.frames_per_second, 3, ' FPS');
  if (text(benchmark.precision)) add('Precision', benchmark.precision.toUpperCase());
  if (numeric(benchmark.num_chunks)) add('Chunks', benchmark.num_chunks);
  if (numeric(benchmark.chunk_size)) add('Chunk size', benchmark.chunk_size);
  if (numeric(benchmark.chunk_overlap)) add('Overlap', benchmark.chunk_overlap);
  addNumber('Valid depth', benchmark.valid_depth_percentage, 0, '%');
  addNumber('Mean confidence', benchmark.mean_confidence, 2);
  addNumber('Median confidence', benchmark.median_confidence, 2);
  addNumber('Peak GPU memory', numeric(benchmark.peak_gpu_memory_mb) ? benchmark.peak_gpu_memory_mb / 1000 : null, 2, ' GB');
  addNumber('GLB', benchmark.glb_size_mb, 2, ' MB');
  addNumber('PLY', benchmark.ply_size_mb, 2, ' MB');
  return rows;
}
