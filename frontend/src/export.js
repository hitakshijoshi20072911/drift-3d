import * as THREE from 'three';
import { TOOLS } from './tools.js';

const LABEL_COLORS = { '': '#ffb224', dim: '#ffb224', note: '#f3c969', good: '#3fb950', bad: '#f2555a' };

export function download(filename, content, type) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function stamp() {
  const d = new Date();
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

/** Canvas snapshot with measurement labels, a north arrow and a scale bar drawn in. */
export function captureImage(viewer, { heading, scale }) {
  viewer.renderer.render(viewer.scene, viewer.camera);
  const src = viewer.renderer.domElement;
  const w = src.width, h = src.height, k = w / (src.clientWidth || 1);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.drawImage(src, 0, 0);
  g.font = `600 ${12 * k}px Segoe UI, system-ui, sans-serif`;
  g.textBaseline = 'middle';
  const pos = new THREE.Vector3();
  viewer.overlay.traverseVisible(o => {
    if (!o.isCSS2DObject) return;
    o.getWorldPosition(pos);
    const p = viewer.project(pos);
    if (!p.visible) return;
    const text = o.element.textContent;
    const tw = g.measureText(text).width + 14 * k;
    const x = p.x * k - tw / 2, y = (p.y - 14) * k - 11 * k;
    g.globalAlpha = o.userData.cls === 'dim' ? 0.6 : 1;
    g.fillStyle = '#0b1015e6';
    g.strokeStyle = LABEL_COLORS[o.userData.cls] ?? LABEL_COLORS[''];
    g.lineWidth = 1.2 * k;
    g.beginPath(); g.roundRect(x, y, tw, 22 * k, 5 * k); g.fill(); g.stroke();
    g.fillStyle = '#fff';
    g.fillText(text, x + 7 * k, y + 11 * k);
    g.globalAlpha = 1;
  });

  // north arrow (bottom-left)
  const cx = 40 * k, cy = h - 40 * k, r = 24 * k;
  g.save();
  g.translate(cx, cy);
  g.fillStyle = '#0b1015d9'; g.strokeStyle = '#33475a'; g.lineWidth = 1.5 * k;
  g.beginPath(); g.arc(0, 0, r, 0, Math.PI * 2); g.fill(); g.stroke();
  g.rotate((-heading * Math.PI) / 180);
  g.fillStyle = '#f2555a';
  g.beginPath(); g.moveTo(0, -r * 0.8); g.lineTo(r * 0.3, 0); g.lineTo(-r * 0.3, 0); g.closePath(); g.fill();
  g.fillStyle = '#c9d4de';
  g.beginPath(); g.moveTo(0, r * 0.8); g.lineTo(r * 0.3, 0); g.lineTo(-r * 0.3, 0); g.closePath(); g.fill();
  g.fillStyle = '#fff'; g.font = `700 ${9 * k}px Segoe UI, sans-serif`; g.textAlign = 'center';
  g.fillText('N', 0, -r * 0.35);
  g.restore();

  // scale bar (bottom-right)
  if (scale) {
    const bw = scale.px * k, bx = w - bw - 24 * k, by = h - 26 * k;
    g.fillStyle = '#0b1015d9';
    g.fillRect(bx - 10 * k, by - 22 * k, bw + 20 * k, 34 * k);
    g.strokeStyle = '#fff'; g.lineWidth = 2 * k;
    g.beginPath(); g.moveTo(bx, by - 6 * k); g.lineTo(bx, by); g.lineTo(bx + bw, by); g.lineTo(bx + bw, by - 6 * k); g.stroke();
    g.fillStyle = '#fff'; g.textAlign = 'center'; g.font = `600 ${12 * k}px Segoe UI, sans-serif`;
    g.fillText(scale.label, bx + bw / 2, by - 13 * k);
  }
  return c.toDataURL('image/png');
}

function lonLatH(geo, p) {
  if (!geo) return [p.x, p.y, p.z];
  const d = geo.describe(p);
  return [Number(d.lon.toFixed(8)), Number(d.lat.toFixed(8)), Number(d.height.toFixed(2))];
}

function geometryOf(result, geo) {
  const coords = result.points.map(p => lonLatH(geo, p));
  if (result.type === 'point' || result.type === 'note') return { type: 'Point', coordinates: coords[0] };
  if (result.type === 'area' || result.type === 'volume') return { type: 'Polygon', coordinates: [[...coords, coords[0]]] };
  return { type: 'LineString', coordinates: coords };
}

function propertiesOf(result) {
  const props = { name: result.name, tool: TOOLS[result.type].name };
  if (result.primary) props.result = `${result.primary.value} ${result.primary.sub ?? ''}`.trim();
  for (const [k, v] of result.values) props[k] = v;
  if (result.coords) Object.assign(props, { MGRS: result.coords.mgrs, UTM: result.coords.utm });
  if (result.note) props.note = result.note;
  return props;
}

export function toGeoJSON(results, geo, dataset) {
  return JSON.stringify({
    type: 'FeatureCollection',
    name: `driftx results - ${dataset}`,
    ...(geo ? {} : { crs_note: 'No georeference.json was loaded: coordinates are local ENU metres.' }),
    features: results.map(r => ({ type: 'Feature', properties: propertiesOf(r), geometry: geometryOf(r, geo) })),
  }, null, 2);
}

const esc = s => String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));

export function toKML(results, geo, dataset) {
  const style = (id, color, fillAlpha = '00') => `<Style id="${id}"><LineStyle><color>ff${color}</color><width>3</width></LineStyle><PolyStyle><color>${fillAlpha}${color}</color></PolyStyle><IconStyle><color>ff${color}</color></IconStyle></Style>`;
  const placemarks = results.map(r => {
    const coords = r.points.map(p => lonLatH(geo, p).join(','));
    const desc = Object.entries(propertiesOf(r)).map(([k, v]) => `<b>${esc(k)}</b>: ${esc(v)}`).join('<br/>');
    let geom;
    if (r.type === 'point' || r.type === 'note') geom = `<Point><coordinates>${coords[0]}</coordinates></Point>`;
    else if (r.type === 'area' || r.type === 'volume') geom = `<Polygon><outerBoundaryIs><LinearRing><coordinates>${[...coords, coords[0]].join(' ')}</coordinates></LinearRing></outerBoundaryIs></Polygon>`;
    else geom = `<LineString><tessellate>1</tessellate><coordinates>${coords.join(' ')}</coordinates></LineString>`;
    const styleId = r.type === 'los' ? (r.visible3d ? 'good' : 'bad') : r.type === 'note' ? 'note' : 'measure';
    return `<Placemark><name>${esc(r.name)}</name><description><![CDATA[${desc}]]></description><styleUrl>#${styleId}</styleUrl>${geom}</Placemark>`;
  });
  return `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>${esc(`driftx - ${dataset}`)}</name>
${style('measure', '24b2ff', '40')}${style('note', 'ffa34f')}${style('good', '50b93f')}${style('bad', '5a55f2')}
${placemarks.join('\n')}
</Document></kml>`;
}

export function toCSV(results, geo) {
  const rows = [['#', 'name', 'tool', 'result', 'details', 'latitude', 'longitude', 'height_m', 'mgrs', 'note']];
  results.forEach((r, i) => {
    const first = r.points[0];
    const d = geo ? geo.describe(first) : null;
    rows.push([
      i + 1, r.name, TOOLS[r.type].name,
      r.primary ? `${r.primary.value} ${r.primary.sub ?? ''}`.trim() : '',
      r.values.map(([k, v]) => `${k}: ${v}`).join('; '),
      d ? d.lat.toFixed(7) : '', d ? d.lon.toFixed(7) : '', d ? d.height.toFixed(2) : first.z.toFixed(2),
      d ? d.mgrs : '', r.note || '',
    ]);
  });
  return rows.map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\r\n');
}

export function toReportHTML({ dataset, image, results, quality, geo }) {
  const rows = results.map((r, i) => {
    const loc = r.coords ?? (geo ? geo.describe(r.points[0]) : null);
    return `<tr><td>${i + 1}</td><td><b>${esc(r.name)}</b><br/><span class="muted">${esc(TOOLS[r.type].name)}</span></td>
<td><b>${esc(r.primary?.value ?? '')}</b> <span class="muted">${esc(r.primary?.sub ?? '')}</span></td>
<td>${r.values.map(([k, v]) => `${esc(k)}: <b>${esc(v)}</b>`).join('<br/>')}${r.note ? `<br/><i>${esc(r.note)}</i>` : ''}</td>
<td class="mono">${loc ? `${esc(loc.latLon)}<br/>${esc(loc.mgrs)}` : ''}</td></tr>`;
  }).join('');
  const qrows = quality.rows.map(q => `<tr><td>${esc(q.label)}</td><td><b>${esc(q.value)}</b></td><td>${esc(q.chipText ?? '')}</td></tr>`).join('');
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>driftx report - ${esc(dataset)}</title>
<style>
body{font:14px/1.5 "Segoe UI",system-ui,sans-serif;color:#18212b;max-width:1000px;margin:32px auto;padding:0 24px}
h1{font-size:22px;margin:0}h2{font-size:15px;margin:28px 0 8px;text-transform:uppercase;letter-spacing:.06em;color:#3b4b5c}
.head{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid #18212b;padding-bottom:10px}
.muted{color:#5d6d7c}.mono{font-family:Consolas,monospace;font-size:12px}
img{width:100%;border-radius:6px;margin-top:16px;border:1px solid #d5dde5}
table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #dde4ea;padding:7px 8px;text-align:left;vertical-align:top}
th{font-size:12px;color:#5d6d7c;text-transform:uppercase;letter-spacing:.04em}
.note{margin-top:14px;padding:10px 12px;background:#eef5ff;border-left:3px solid #2f86f0;font-size:13px}
@media print{body{margin:0}img{break-inside:avoid}}
</style></head><body>
<div class="head"><div><h1>3D analysis report</h1><div class="muted">${esc(dataset)}</div></div>
<div class="muted">driftx · ${esc(new Date().toLocaleString())}</div></div>
<img src="${image}" alt="Model view with measurements"/>
<h2>Results</h2>
<table><tr><th>#</th><th>Item</th><th>Result</th><th>Details</th><th>Location</th></tr>${rows || '<tr><td colspan="5" class="muted">No measurements.</td></tr>'}</table>
<h2>Model quality</h2>
<table>${qrows}</table>
<div class="note">${esc(quality.note)}</div>
</body></html>`;
}
