import { fmt } from './geo.js';
import { findKey } from './data.js';

const num = v => (v === null || v === undefined || v === '' ? NaN : Number(v));

function grade(value, good, warn, higherIsBetter = true) {
  if (!Number.isFinite(value)) return 'info';
  if (higherIsBetter) return value >= good ? 'good' : value >= warn ? 'warn' : 'bad';
  return value <= good ? 'good' : value <= warn ? 'warn' : 'bad';
}

const CHIP_TEXT = { good: 'Good', warn: 'Check', bad: 'Low', info: '' };

/** Plain-language quality summary from the pipeline's report files. */
export function summarizeQuality(meta, { mesh, cloud }) {
  const run = meta.run_report ?? {};
  const ver = meta.verification_report ?? {};
  const vm = meta.viewer_metadata ?? {};
  const georef = meta.georeference ?? {};

  const seconds = num(run.wall_clock_seconds ?? vm.processing_seconds);
  const sparse = run.sparse_metrics ?? vm.sparse_metrics ?? {};
  const regPct = num(sparse.registration_percent ?? vm.registration_percent);
  const registered = num(sparse.registered_images);
  const reproj = num(sparse.mean_reprojection_error_px ?? findKey(ver, 'mean_reprojection_error_px'));
  const validation = run.validation ?? vm.validation ?? {};
  const gpsRmse = num(validation.gps_alignment_rmse_m);
  const footprint = findKey(ver, 'footprint_coverage');
  const coverage = num(footprint?.fraction);
  const points = num(run.products?.point_count ?? vm.products?.point_count) || (cloud ? cloud.geometry.attributes.position.count : NaN);
  const faces = mesh ? (mesh.geometry.index ? mesh.geometry.index.count / 3 : mesh.geometry.attributes.position.count / 3) : NaN;
  const checkpoint = validation.surveyed_checkpoints ?? vm.validation?.surveyed_checkpoints;
  const checkpointRmse = num(checkpoint?.rmse_3d_m);
  const vertical = georef.vertical_reference ?? vm.vertical_reference ?? null;

  const rows = [];
  const add = (label, value, chip, hint) => rows.push({ label, value, chip, chipText: CHIP_TEXT[chip], hint });
  if (Number.isFinite(seconds)) add('Processing time', fmt.duration(seconds), 'info', 'Wall-clock time of the pipeline run');
  if (Number.isFinite(regPct)) add('Video frames used', `${Number.isFinite(registered) ? `${registered} · ` : ''}${regPct.toFixed(1)}%`, grade(regPct, 90, 70), 'Share of selected frames placed in the 3D model');
  if (Number.isFinite(reproj)) add('Reprojection error', `${reproj.toFixed(2)} px`, grade(reproj, 1, 2, false), 'How closely the 3D model matches the photos (lower is better)');
  if (Number.isFinite(coverage)) add('Ground coverage', fmt.percent(coverage), grade(coverage, 0.75, 0.5), 'Share of the flown area with surface data');
  if (Number.isFinite(gpsRmse)) add('Offset vs drone GPS', `${gpsRmse.toFixed(2)} m`, grade(gpsRmse, 1, 3, false), 'RMSE between the model camera path and the drone GPS');
  if (Number.isFinite(checkpointRmse)) add('Surveyed checkpoint error', `${checkpointRmse.toFixed(2)} m`, grade(checkpointRmse, 1, 2, false), 'Independent accuracy against surveyed points');
  else add('Surveyed accuracy', 'Not checked', 'info', 'No surveyed checkpoints were supplied for this run');
  if (Number.isFinite(points)) add('Points in cloud', Math.round(points).toLocaleString(), 'info');
  if (Number.isFinite(faces)) add('Mesh faces', Math.round(faces).toLocaleString(), 'info', mesh?.userData.textured ? 'Photo-textured' : 'Untextured');

  const state = ver.terminal_state ?? null;
  let status;
  if (state === 'PRODUCTION_READY' || ver.production_ready === true) status = { cls: 'good', title: 'Ready for use', text: 'All quality checks passed.' };
  else if (state === 'PARTIAL_VALID') status = { cls: 'warn', title: 'Usable, with open checks', text: 'The model is complete; some accuracy or timing targets are not yet proven.' };
  else if (state) status = { cls: 'bad', title: 'Not reliable', text: `Pipeline state: ${state}` };
  else if (Object.keys(ver).length) {
    const open = Object.entries(ver.quality_checks ?? {}).filter(([, v]) => v === false).map(([k]) => k.replace(/_/g, ' '));
    status = open.length
      ? { cls: 'warn', title: `${open.length} quality check${open.length > 1 ? 's' : ''} open`, text: open.slice(0, 3).join(', ') + (open.length > 3 ? ', …' : '') }
      : { cls: 'good', title: 'All quality checks passed', text: 'From verification_report.json.' };
  } else status = { cls: 'info', title: 'Quality report not loaded', text: 'Open run_report.json and verification_report.json for quality details.' };

  const offset = Number.isFinite(checkpointRmse) ? checkpointRmse : gpsRmse;
  const note = Number.isFinite(offset)
    ? `Distances, heights, areas and volumes are measured inside one consistent 3D model. Absolute coordinates (latitude/longitude, MGRS) may be offset by about ${offset.toFixed(1)} m${Number.isFinite(checkpointRmse) ? ' (surveyed checkpoints)' : ' (alignment to the drone GPS)'}.`
    : 'Distances, heights, areas and volumes are measured inside one consistent 3D model. Absolute position accuracy is unknown for this run.';

  return {
    status,
    rows,
    note,
    heightNote: vertical?.message ?? null,
    crs: georef.projected_crs ?? vm.crs ?? null,
  };
}
