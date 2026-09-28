import * as THREE from 'three';
import { Viewer, RAMP_CSS } from './scene.js';
import { HeightField } from './terrain.js';
import { GeoFrame, fmt } from './geo.js';
import { classifyFiles, filesFromDrop, loadReconstruction } from './data.js';
import { TOOLS, buildResult, liveReadout, previewGroup, disposeGroup, markers, setLineResolution } from './tools.js';
import { ProfileChart } from './chart.js';
import { summarizeQuality } from './quality.js';
import { captureImage, download, stamp, toGeoJSON, toKML, toCSV, toReportHTML } from './export.js';

const $ = sel => document.querySelector(sel);
const $$ = sel => [...document.querySelectorAll(sel)];
const esc = s => String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));

const viewer = new Viewer($('#canvas-host'));
viewer.onResize = [(w, h) => setLineResolution(w, h)];
setLineResolution(viewer.host.clientWidth || innerWidth, viewer.host.clientHeight || innerHeight);

const app = {
  terrain: null,
  geo: null,
  dataset: '',
  quality: null,
  results: [],
  selected: null,
  tool: null,
  points: [],
  cursor: null,
  preview: null,
  options: { baseMode: 'fit', observerHeight: 1.8, targetHeight: 1.0 },
};

// ---------------------------------------------------------------- helpers

let toastTimer;
function toast(message, isError = false) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.toggle('error', isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 3200);
}

const nextFrame = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));

function progress(stage, fraction) {
  $('#loading-detail').textContent = stage;
  $('#loading-bar').style.width = `${Math.round(fraction * 100)}%`;
}

function toolIcon(type) {
  return document.querySelector(`#toolbar [data-tool="${type}"] svg`)?.outerHTML ?? '';
}

const heightOf = z => (app.geo ? app.geo.height(z) : z);

// ---------------------------------------------------------------- loading

async function loadSource(source, datasetName) {
  if (!source.model && !source.cloud) {
    toast('No model.glb, mesh.ply or point_cloud.ply found in what you opened.', true);
    return;
  }
  $('#loading').hidden = false;
  $('#loading-title').textContent = `Opening ${datasetName || 'reconstruction'}`;
  progress('Starting', 0);
  try {
    const { mesh, cloud, meta } = await loadReconstruction(source, progress);
    if (!mesh && !cloud) throw new Error('The model file could not be read.');
    resetResults();
    viewer.setModel({ mesh, cloud });

    progress('Building the analysis surface', 0.96);
    await nextFrame();
    const terrain = HeightField.forBox(viewer.box);
    if (mesh) {
      const g = mesh.geometry;
      terrain.addTriangles(g.attributes.position.array, g.index ? g.index.array : null);
    }
    if (cloud) terrain.addPoints(cloud.geometry.attributes.position.array);
    terrain.fillPinholes();
    app.terrain = terrain;
    viewer.terrain = terrain;

    const origin = meta.georeference?.origin ?? meta.viewer_metadata?.origin;
    app.geo = origin ? new GeoFrame(origin, meta.georeference?.vertical_reference ?? null) : null;
    app.dataset = datasetName || 'Reconstruction';
    app.quality = summarizeQuality(meta, { mesh, cloud });

    const [lo, hi] = terrain.percentiles();
    viewer.setHeightRange(lo, hi);
    app.geo?.setGround(lo);
    $('#legend-min').textContent = fmt.height(heightOf(lo));
    $('#legend-max').textContent = fmt.height(heightOf(hi));
    $('#legend-title').textContent = app.geo ? app.geo.heightLabel : 'Height';
    document.querySelector('#legend .legend-bar').style.background = RAMP_CSS;

    $('[data-display="photo"]').disabled = !mesh && !cloud?.geometry.attributes.color;
    $('[data-display="points"]').disabled = !cloud;
    setDisplay(mesh ? 'photo' : cloud?.geometry.attributes.color ? 'photo' : 'height');

    renderQuality();
    showWorkspace();
    viewer.fit();
    history.pushState({ view: 'analysis' }, '', location.pathname + location.search + '#analysis');
    toast(app.geo ? 'Model ready. Pick a tool on the left.' : 'Model ready. No georeference.json: coordinates are local metres.');
  } catch (err) {
    console.error(err);
    toast(`Could not open the reconstruction: ${err.message}`, true);
  } finally {
    $('#loading').hidden = true;
  }
}

function showLanding() {
  deactivateTool();
  closeProfile();
  document.body.classList.remove('has-model');
  for (const id of ['#toolbar', '#panel', '#statusbar', '#view-controls', '#export-menu', '#btn-screenshot', '#hud', '#dataset', '#btn-open']) $(id).hidden = true;
  $('#landing').hidden = false;
  const back = $('#btn-resume');
  back.hidden = !viewer.box;
  if (viewer.box) back.textContent = `Back to ${app.dataset}`;
}

addEventListener('popstate', e => {
  if (e.state?.view === 'analysis' && viewer.box) showWorkspace(); else showLanding();
});

function showWorkspace() {
  $('#btn-open').hidden = false;
  document.body.classList.add('has-model');
  for (const id of ['#toolbar', '#panel', '#statusbar', '#view-controls', '#export-menu', '#btn-screenshot', '#hud']) $(id).hidden = false;
  $('#landing').hidden = true;
  const ds = $('#dataset');
  ds.hidden = false;
  ds.textContent = app.dataset;
  ds.title = app.quality.crs ? `${app.dataset} · ${app.quality.crs}` : app.dataset;
  requestAnimationFrame(() => viewer.resize());
}

async function openFiles(files) {
  $('#data-credit').hidden = true;
  const picked = classifyFiles(files);
  const name = picked.folder || (picked.model ?? picked.cloud)?.name?.replace(/\.[^.]+$/, '') || 'Reconstruction';
  await loadSource(picked, name);
}

async function openUrlFolder(base, name) {
  const b = base.endsWith('/') ? base : `${base}/`;
  const exists = async f => {
    const r = await fetch(b + f, { method: 'HEAD' }).catch(() => null);
    return Boolean(r?.ok) && !(r.headers.get('content-type') || '').includes('text/html');
  };
  const json = {};
  for (const f of ['viewer_metadata.json', 'run_report.json', 'verification_report.json', 'georeference.json']) {
    if (await exists(f)) json[f] = b + f;
  }
  const model = (await exists('model.glb')) ? `${b}model.glb` : (await exists('mesh.ply')) ? `${b}mesh.ply` : null;
  const cloud = (await exists('point_cloud.ply')) ? `${b}point_cloud.ply` : null;
  await loadSource({ model, cloud, json }, name);
}

$('#btn-open').addEventListener('click', () => (history.state?.view === 'analysis' ? history.back() : showLanding()));
$('#btn-resume').addEventListener('click', () => (history.state?.view === 'analysis' ? showWorkspace() : history.forward()));
$('#folder-input').addEventListener('change', e => { const f = [...e.target.files]; e.target.value = ''; if (f.length) openFiles(f); });
$('#files-input').addEventListener('change', e => { const f = [...e.target.files]; e.target.value = ''; if (f.length) openFiles(f); });

let dragDepth = 0;
addEventListener('dragenter', e => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', async e => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  const files = await filesFromDrop(e.dataTransfer);
  if (files.length) openFiles(files);
});

// Demo data (e.g. when hosted with a demo/ folder) and ?data=<folder-url>
(async () => {
  const params = new URLSearchParams(location.search);
  const base = import.meta.env.BASE_URL;
  if (params.get('data')) return openUrlFolder(params.get('data'), params.get('name') || 'Reconstruction');
  const catalog = await fetch(`${base}demo/index.json`).then(r => (r.ok ? r.json() : null)).catch(() => null);
  if (catalog?.demos?.length) {
    const list = $('#demo-list');
    list.hidden = false;
    for (const demo of catalog.demos) {
      const button = document.createElement('button');
      button.className = 'btn demo-button';
      button.innerHTML = `<strong>${esc(demo.name)}</strong><small>PRECOMPUTED DEMO · ${esc(demo.id)}</small>`;
      button.addEventListener('click', () => openUrlFolder(`${base}demo/${demo.path}`, demo.name));
      list.appendChild(button);
    }
    if (params.get('demo')) {
      const selected = catalog.demos.find(d => d.id === params.get('demo')) || catalog.demos[0];
      openUrlFolder(`${base}demo/${selected.path}`, selected.name);
    }
  }
})();

// ---------------------------------------------------------------- display

function setDisplay(mode) {
  if ($(`[data-display="${mode}"]`)?.disabled) return;
  viewer.setDisplay(mode);
  $$('[data-display]').forEach(b => b.classList.toggle('active', b.dataset.display === mode));
  $('#legend').hidden = mode !== 'height';
}
$$('[data-display]').forEach(b => b.addEventListener('click', () => setDisplay(b.dataset.display)));
$('#btn-top').addEventListener('click', () => viewer.topView());
$('#btn-fit').addEventListener('click', () => viewer.fit());
$('#compass').addEventListener('click', () => viewer.topView());

const compassSvg = $('#compass svg');
let frameCount = 0;
viewer.onFrame.push(() => {
  if (!viewer.box) return;
  compassSvg.style.transform = `rotate(${-viewer.heading()}deg)`;
  if (frameCount++ % 6 === 0) updateScaleBar();
});

function niceScale(maxMetres) {
  const pow = 10 ** Math.floor(Math.log10(maxMetres));
  for (const m of [5, 2, 1]) if (m * pow <= maxMetres) return m * pow;
  return pow;
}
let scaleState = null;
function updateScaleBar() {
  const mpp = viewer.metresPerPixel();
  if (!Number.isFinite(mpp) || mpp <= 0) return;
  const metres = niceScale(mpp * 120);
  const px = metres / mpp;
  const labelText = metres >= 1000 ? `${metres / 1000} km` : `${metres} m`;
  scaleState = { px, label: labelText };
  $('.scalebar-line').style.width = `${px}px`;
  $('#scalebar-label').textContent = labelText;
}

// ---------------------------------------------------------------- pointer

const stage = $('#stage');
const canvas = viewer.renderer.domElement;
let down = null;
let moveQueued = false, lastMove = null;

canvas.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY, button: e.button }; });
canvas.addEventListener('pointerup', e => {
  if (!down || e.button !== 0 || down.button !== 0) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  down = null;
  if (moved < 5) handleClick(e);
});
canvas.addEventListener('pointermove', e => {
  lastMove = e;
  if (moveQueued) return;
  moveQueued = true;
  requestAnimationFrame(() => { moveQueued = false; handleMove(lastMove); });
});
canvas.addEventListener('pointerleave', () => { app.cursor = null; refreshPreview(); });
canvas.addEventListener('dblclick', e => {
  if (!app.tool) return;
  const kind = TOOLS[app.tool].kind;
  if (kind !== 'path' && kind !== 'polygon') return;
  e.preventDefault();
  const n = app.points.length;
  if (n >= 2 && app.points[n - 1].distanceTo(app.points[n - 2]) < viewer.metresPerPixel() * 6) app.points.pop();
  finishTool();
});

function handleMove(e) {
  if (!viewer.box) return;
  const p = viewer.pick(e.clientX, e.clientY);
  updateCoords(p);
  if (app.tool) { app.cursor = p; refreshPreview(); }
}

function updateCoords(p) {
  const el = $('#status-coords');
  if (!p) { el.innerHTML = '<span class="muted">Move over the model to read coordinates</span>'; return; }
  if (app.geo) {
    const d = app.geo.describe(p);
    el.innerHTML = `<span><span class="k">X, Y, Z</span><span class="v">${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)} m</span></span>`
      + `<span><span class="k">Lat/Lon</span><span class="v">${d.latLon}</span></span>`
      + `<span><span class="k">MGRS</span><span class="v">${d.mgrs}</span></span>`
      + `<span><span class="k">${esc(app.geo.heightLabel)}</span><span class="v">${d.height.toFixed(1)} m</span></span>`;
  } else {
    el.innerHTML = `<span><span class="k">E / N / Up</span><span class="v">${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)} m (local)</span></span>`;
  }
}

function handleClick(e) {
  if (!app.tool) return;
  const p = viewer.pick(e.clientX, e.clientY);
  if (!p) { toast('Click on the model surface.'); return; }
  app.points.push(p);
  const kind = TOOLS[app.tool].kind;
  if (kind === 'single' || (kind === 'pair' && app.points.length === 2)) finishTool();
  else refreshPreview();
}

// ---------------------------------------------------------------- tools

function activateTool(type) {
  if (app.tool === type) { deactivateTool(); return; }
  clearPreview();
  app.tool = type;
  app.points = [];
  const def = TOOLS[type];
  $$('#toolbar [data-tool]').forEach(b => b.classList.toggle('active', b.dataset.tool === type));
  stage.classList.add('picking');
  $('#tool-card').hidden = false;
  $('#tool-card-title').textContent = def.name;
  $('#tool-card-help').textContent = def.help;
  $('#tool-card-actions').hidden = !(def.kind === 'path' || def.kind === 'polygon');
  renderToolOptions(type);
  refreshPreview();
}

function deactivateTool() {
  clearPreview();
  app.tool = null;
  app.points = [];
  app.cursor = null;
  $$('#toolbar [data-tool]').forEach(b => b.classList.remove('active'));
  stage.classList.remove('picking');
  $('#tool-card').hidden = true;
  $('#tool-hint').hidden = true;
}

function renderToolOptions(type) {
  const box = $('#tool-card-options');
  box.innerHTML = '';
  if (type === 'volume') {
    box.innerHTML = `<div class="option"><span>Base under the pile</span><div class="segmented" id="base-mode">
      <button data-base="fit" class="${app.options.baseMode === 'fit' ? 'active' : ''}" title="Plane fitted to the ground along the outline">Fitted</button>
      <button data-base="lowest" class="${app.options.baseMode === 'lowest' ? 'active' : ''}" title="Flat base at the lowest point of the outline">Lowest</button></div></div>`;
    box.querySelectorAll('[data-base]').forEach(b => b.addEventListener('click', () => {
      app.options.baseMode = b.dataset.base;
      box.querySelectorAll('[data-base]').forEach(x => x.classList.toggle('active', x === b));
    }));
  } else if (type === 'los') {
    box.innerHTML = `<label class="option">Observer above surface (m) <input type="number" id="opt-observer" step="0.1" min="0" value="${app.options.observerHeight}"></label>
      <label class="option">Target above surface (m) <input type="number" id="opt-target" step="0.1" min="0" value="${app.options.targetHeight}"></label>`;
    $('#opt-observer').addEventListener('input', e => { app.options.observerHeight = Math.max(0, Number(e.target.value) || 0); });
    $('#opt-target').addEventListener('input', e => { app.options.targetHeight = Math.max(0, Number(e.target.value) || 0); });
  }
}

function hintText() {
  const t = app.tool, n = app.points.length, kind = TOOLS[t].kind;
  if (kind === 'single') return t === 'note' ? 'Click the location to mark' : 'Click a location';
  if (t === 'height') return n === 0 ? 'Click the <b>base</b> (e.g. the ground)' : 'Click the <b>top</b>';
  if (t === 'los') return n === 0 ? 'Click the <b>observer</b> position' : 'Click the <b>target</b>';
  if (kind === 'polygon') return n < 3 ? `Click the corners of the outline (${n}/3)` : 'Click more corners · <b>Enter</b> or double-click to close';
  return n === 0 ? 'Click the start point' : n === 1 ? 'Click the next point' : 'Click more points · <b>Enter</b> or double-click to finish';
}

function clearPreview() {
  if (app.preview) { disposeGroup(app.preview); app.preview = null; }
}

function refreshPreview() {
  clearPreview();
  if (!app.tool) return;
  const hint = $('#tool-hint');
  hint.hidden = false;
  hint.innerHTML = `${esc(TOOLS[app.tool].name)} · ${hintText()} · <b>Esc</b> cancel`;
  if (app.points.length || app.cursor) {
    app.preview = previewGroup(app.tool, app.points, app.points.length ? app.cursor : null);
    viewer.overlay.add(app.preview);
  }
  const live = liveReadout(app.tool, app.points, app.cursor);
  $('#tool-card-live').innerHTML = live ? `<div class="live-value">${live.value}</div><div class="live-sub">${esc(live.sub)}</div>` : '';
  $('#tool-finish').disabled = app.points.length < (TOOLS[app.tool].min ?? 1);
}

function finishTool() {
  if (!app.tool) return;
  const def = TOOLS[app.tool];
  if (app.points.length < (def.min ?? 1)) {
    toast(def.kind === 'polygon' ? 'An outline needs at least 3 corners.' : 'Place at least 2 points first.');
    return;
  }
  let result;
  try {
    result = buildResult(app.tool, app.points, { terrain: app.terrain, geo: app.geo, viewer, options: { ...app.options } });
  } catch (err) {
    console.error(err);
    toast(`Could not compute this ${def.name.toLowerCase()}: ${err.message}`, true);
    return;
  }
  const count = app.results.filter(r => r.type === app.tool).length + 1;
  result.id = crypto.randomUUID?.() ?? String(Date.now() + Math.random());
  result.name = `${def.name} ${count}`;
  if (result.type === 'note') result.labelObj.element.textContent = result.name;
  app.results.push(result);
  viewer.overlay.add(result.group);
  app.points = [];
  clearPreview();
  refreshPreview();
  switchTab('results');
  renderResults();
  select(result);
  if (result.type === 'note') {
    const input = document.querySelector(`[data-id="${result.id}"] .result-name`);
    input?.focus(); input?.select();
  }
}

$$('#toolbar [data-tool]').forEach(b => b.addEventListener('click', () => activateTool(b.dataset.tool)));
$('#tool-cancel').addEventListener('click', () => (app.points.length ? (app.points = [], refreshPreview()) : deactivateTool()));
$('#tool-undo').addEventListener('click', () => { app.points.pop(); refreshPreview(); });
$('#tool-finish').addEventListener('click', finishTool);

// ---------------------------------------------------------------- results

function resetResults() {
  for (const r of app.results) disposeGroup(r.group);
  app.results = [];
  app.selected = null;
  closeProfile();
  renderResults();
  deactivateTool();
}

function renderResults() {
  const list = $('#results-list');
  $('#results-count').textContent = app.results.length;
  $('#results-empty').hidden = app.results.length > 0;
  list.innerHTML = app.results.map((r, i) => {
    const values = r.values.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    const coords = r.coords ? `<div class="coords">${esc(r.coords.latLon)}<br>MGRS ${esc(r.coords.mgrs)}<br>UTM ${esc(r.coords.utm)}</div>` : '';
    const primary = r.primary
      ? `<div class="result-primary">${r.primary.verdict ? `<span class="verdict ${r.primary.verdict}">${esc(r.primary.value)}</span>` : esc(r.primary.value)}<small>${esc(r.primary.sub ?? '')}</small></div>`
      : '';
    return `<li class="result ${app.selected === r ? 'selected' : ''} ${r.group.visible ? '' : 'hidden-result'}" data-id="${r.id}" data-index="${i}">
      <div class="result-head">${toolIcon(r.type)}
        <input class="result-name" value="${esc(r.name)}" aria-label="Name" spellcheck="false">
        <div class="result-actions">
          <button data-act="zoom" title="Zoom to"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5M11 8v6M8 11h6"/></svg></button>
          <button data-act="toggle" title="${r.group.visible ? 'Hide' : 'Show'}"><svg viewBox="0 0 24 24">${r.group.visible ? '<path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z"/><circle cx="12" cy="12" r="2.8"/>' : '<path d="M3 3l18 18M10.6 6.1A10 10 0 0 1 12 6c6.5 0 10 6 10 6a17 17 0 0 1-3.2 3.8M6.1 7.9C3.6 9.6 2 12 2 12s3.5 6 10 6a9.7 9.7 0 0 0 4-.9"/>'}</svg></button>
          <button data-act="delete" title="Delete"><svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/></svg></button>
        </div>
      </div>
      ${primary}
      ${values ? `<dl class="result-values">${values}</dl>` : ''}
      ${coords}
      ${r.note ? `<div class="result-note">${esc(r.note)}</div>` : ''}
    </li>`;
  }).join('');
}

function resultById(id) { return app.results.find(r => r.id === id); }

function select(r) {
  app.selected = r;
  $$('#results-list .result').forEach(li => li.classList.toggle('selected', li.dataset.id === r?.id));
  if (r?.type === 'profile') openProfile(r); else if (r) closeProfile();
}

function zoomTo(r) {
  const box = new THREE.Box3().setFromPoints(r.points);
  const size = box.getSize(new THREE.Vector3());
  box.expandByScalar(Math.max(15, Math.max(size.x, size.y) * 0.25));
  viewer.frameBox(box, { padding: 1.2 });
}

$('#results-list').addEventListener('click', e => {
  const li = e.target.closest('.result');
  if (!li) return;
  const r = resultById(li.dataset.id);
  const act = e.target.closest('[data-act]')?.dataset.act;
  if (act === 'zoom') zoomTo(r);
  else if (act === 'toggle') { r.group.visible = !r.group.visible; renderResults(); }
  else if (act === 'delete') {
    disposeGroup(r.group);
    app.results = app.results.filter(x => x !== r);
    if (app.selected === r) { app.selected = null; closeProfile(); }
    renderResults();
    return;
  }
  if (!e.target.classList.contains('result-name')) select(r);
});
$('#results-list').addEventListener('input', e => {
  if (!e.target.classList.contains('result-name')) return;
  const r = resultById(e.target.closest('.result').dataset.id);
  r.name = e.target.value;
  if (r.type === 'note') r.labelObj.element.textContent = r.name || 'Note';
});
$('#results-list').addEventListener('keydown', e => {
  if (e.target.classList.contains('result-name') && e.key === 'Enter') e.target.blur();
});

// ---------------------------------------------------------------- profile drawer

let profileMarker = null;
const chart = new ProfileChart($('#profile-canvas'), idx => {
  if (profileMarker) { disposeGroup(profileMarker); profileMarker = null; }
  const r = app.selected;
  if (idx < 0 || !r?.profile) return;
  const pr = r.profile;
  profileMarker = new THREE.Group();
  profileMarker.add(markers([new THREE.Vector3(pr.x[idx], pr.y[idx], pr.z[idx])], '#ffffff', 14));
  viewer.overlay.add(profileMarker);
});

function openProfile(r) {
  $('#profile-drawer').hidden = false;
  $('#profile-title').textContent = `${r.name} · surface profile`;
  $('#profile-summary').textContent = `${fmt.length(r.profile.length)} · climb +${fmt.height(r.profile.gain)} · average grade ${fmt.slope(r.profile.avgGrade)}`;
  chart.set(r.profile, heightOf);
}
function closeProfile() {
  $('#profile-drawer').hidden = true;
  if (profileMarker) { disposeGroup(profileMarker); profileMarker = null; }
}
$('#profile-close').addEventListener('click', closeProfile);

// ---------------------------------------------------------------- quality tab

function renderQuality() {
  const q = app.quality;
  const chip = (cls, text) => (text ? `<span class="chip ${cls}">${esc(text)}</span>` : '');
  $('#quality-body').innerHTML = `
    <div class="q-status"><span class="chip ${q.status.cls}">${q.status.cls === 'good' ? 'Pass' : q.status.cls === 'warn' ? 'Open checks' : q.status.cls === 'bad' ? 'Fail' : 'Info'}</span>
      <div><b>${esc(q.status.title)}</b><small>${esc(q.status.text)}</small></div></div>
    <div class="q-rows">${q.rows.map(r => `<div class="q-row" title="${esc(r.hint ?? '')}"><span class="q-label">${esc(r.label)}</span><span class="q-value">${esc(r.value)}</span>${chip(r.chip, r.chipText) || '<span></span>'}</div>`).join('')}</div>
    <div class="q-note"><b>What this means for your measurements.</b> ${esc(q.note)}</div>
    ${q.heightNote ? `<div class="q-section">Heights</div><p class="muted">${esc(q.heightNote)}</p>` : ''}
    ${q.crs ? `<div class="q-section">Coordinate system</div><p class="muted">Exports use WGS84 latitude/longitude. The pipeline's GIS products use ${esc(q.crs)}.</p>` : ''}`;
}

function switchTab(name) {
  $$('.tabs [data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('[data-tab-body]').forEach(s => { s.hidden = s.dataset.tabBody !== name; });
}
$$('.tabs [data-tab]').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));

// ---------------------------------------------------------------- export

const exportMenu = $('#export-menu .menu-list');
$('#btn-export').addEventListener('click', e => { e.stopPropagation(); exportMenu.hidden = !exportMenu.hidden; });
addEventListener('click', e => { if (!e.target.closest('#export-menu')) exportMenu.hidden = true; });

function screenshotData() {
  updateScaleBar();
  return captureImage(viewer, { heading: viewer.heading(), scale: scaleState });
}

$('#btn-screenshot').addEventListener('click', () => {
  const url = screenshotData();
  fetch(url).then(r => r.blob()).then(blob => download(`driftx-${slug()}-${stamp()}.png`, blob));
  toast('Screenshot saved.');
});

const slug = () => app.dataset.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'model';

exportMenu.addEventListener('click', e => {
  const kind = e.target.closest('[data-export]')?.dataset.export;
  if (!kind) return;
  exportMenu.hidden = true;
  if (kind !== 'report' && !app.results.length) { toast('No results yet. Measure something first.'); return; }
  const base = `driftx-${slug()}-${stamp()}`;
  if (kind === 'geojson') download(`${base}.geojson`, toGeoJSON(app.results, app.geo, app.dataset), 'application/geo+json');
  if (kind === 'kml') download(`${base}.kml`, toKML(app.results, app.geo, app.dataset), 'application/vnd.google-earth.kml+xml');
  if (kind === 'csv') download(`${base}.csv`, toCSV(app.results, app.geo), 'text/csv');
  if (kind === 'report') {
    const html = toReportHTML({ dataset: app.dataset, image: screenshotData(), results: app.results, quality: app.quality, geo: app.geo });
    download(`${base}-report.html`, html, 'text/html');
  }
  toast(`${kind === 'report' ? 'Report' : kind.toUpperCase()} exported.`);
});

// ---------------------------------------------------------------- help & keys

$('#btn-help').addEventListener('click', () => { $('#help').hidden = false; });
$('#help-close').addEventListener('click', () => { $('#help').hidden = true; });
$('#help').addEventListener('click', e => { if (e.target.id === 'help') $('#help').hidden = true; });

const KEY_TOOLS = Object.fromEntries(Object.entries(TOOLS).map(([id, def]) => [def.key, id]));

addEventListener('keydown', e => {
  const typing = e.target.matches('input, textarea, [contenteditable]');
  if (e.key === 'Escape') {
    if (typing) { e.target.blur(); return; }
    if (!$('#help').hidden) { $('#help').hidden = true; return; }
    if (!exportMenu.hidden) { exportMenu.hidden = true; return; }
    if (app.tool && app.points.length) { app.points = []; refreshPreview(); return; }
    if (app.tool) { deactivateTool(); return; }
    closeProfile();
    return;
  }
  if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === '?') { $('#help').hidden = !$('#help').hidden; return; }
  if (!viewer.box) return;
  if (e.key === 'Enter' && app.tool) { e.preventDefault(); finishTool(); return; }
  if (e.key === 'Backspace' && app.tool && app.points.length) { e.preventDefault(); app.points.pop(); refreshPreview(); return; }
  const k = e.key.toLowerCase();
  if (KEY_TOOLS[k]) { activateTool(KEY_TOOLS[k]); return; }
  if (k === 't') viewer.topView();
  else if (k === 'f') viewer.fit();
  else if (k === '1') setDisplay('photo');
  else if (k === '2') setDisplay('height');
  else if (k === '3') setDisplay('points');
});

// Development-only handle for automated checks in the browser console.
if (import.meta.env.DEV) window.__driftx = { app, viewer };
