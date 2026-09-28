import * as THREE from 'three';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { fmt } from './geo.js';
import { polygonArea, pathLength } from './terrain.js';

export const COLORS = { measure: '#ffb224', accent: '#f3c969', good: '#3fb950', bad: '#f2555a', white: '#ffffff' };

export const TOOLS = {
  point: { name: 'Point', key: 'i', kind: 'single', help: 'Click any location to read its latitude/longitude, MGRS grid reference, UTM and height.' },
  distance: { name: 'Distance', key: 'd', kind: 'path', min: 2, help: 'Click points along the route or span. Double-click or press Enter to finish.' },
  height: { name: 'Height', key: 'h', kind: 'pair', help: 'Click the base (for example the ground), then the top of the structure.' },
  area: { name: 'Area', key: 'a', kind: 'polygon', min: 3, help: 'Click around the outline. Double-click or press Enter to close the shape.' },
  volume: { name: 'Volume', key: 'v', kind: 'polygon', min: 3, help: 'Outline the stockpile, debris or pit on the ground just around it. The base is fitted to that outline.' },
  profile: { name: 'Profile', key: 'p', kind: 'path', min: 2, help: 'Click points along the road, ridge or river. Press Enter to see the elevation chart.' },
  los: { name: 'Sight line', key: 'l', kind: 'pair', help: 'Click the observer position, then the target. Heights above the surface are set below.' },
  note: { name: 'Note', key: 'n', kind: 'single', help: 'Click the location to mark. Type the label in the result card on the right.' },
};

// ---------------- drawing helpers ----------------

let dotTexture = null;
function dot() {
  if (dotTexture) return dotTexture;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.beginPath(); g.arc(32, 32, 26, 0, Math.PI * 2);
  g.fillStyle = '#fff'; g.fill();
  g.lineWidth = 8; g.strokeStyle = '#0b1015'; g.stroke();
  dotTexture = new THREE.CanvasTexture(c);
  return dotTexture;
}

export const lineMaterials = new Set();
let resolution = [innerWidth, innerHeight];

/** Keep fat-line widths in screen pixels when the canvas size changes. */
export function setLineResolution(w, h) {
  resolution = [w, h];
  for (const m of lineMaterials) m.resolution.set(w, h);
}

export function fatLine(points, color, { width = 3, dashed = false, opacity = 1 } = {}) {
  const g = new LineGeometry();
  g.setPositions(points.flatMap(p => [p.x, p.y, p.z]));
  const m = new LineMaterial({ color, linewidth: width, depthTest: false, transparent: true, opacity, dashed, dashSize: 2, gapSize: 1.2 });
  m.resolution.set(resolution[0], resolution[1]);
  lineMaterials.add(m);
  const line = new Line2(g, m);
  line.computeLineDistances();
  line.renderOrder = 998;
  line.userData.material = m;
  return line;
}

export function markers(points, color, size = 11) {
  const g = new THREE.BufferGeometry().setFromPoints(points);
  const m = new THREE.PointsMaterial({ color, size, sizeAttenuation: false, map: dot(), transparent: true, alphaTest: 0.3, depthTest: false });
  const p = new THREE.Points(g, m);
  p.renderOrder = 1000;
  return p;
}

export function label(text, position, cls = '') {
  const el = document.createElement('div');
  el.className = `m-label ${cls}`;
  el.textContent = text;
  const obj = new CSS2DObject(el);
  obj.position.copy(position);
  obj.userData.text = text;
  obj.userData.cls = cls;
  return obj;
}

function fill(points, color, opacity = 0.18) {
  const contour = points.map(p => new THREE.Vector2(p.x, p.y));
  const tris = THREE.ShapeUtils.triangulateShape(contour, []);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(points.flatMap(p => [p.x, p.y, p.z + 0.05]), 3));
  g.setIndex(tris.flat());
  const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(g, m);
  mesh.renderOrder = 997;
  return mesh;
}

export function disposeGroup(group) {
  group.traverse(o => {
    o.geometry?.dispose?.();
    if (o.material) {
      lineMaterials.delete(o.material);
      o.material.dispose?.();
    }
    if (o.isCSS2DObject) o.element.remove();
  });
  group.removeFromParent();
}

const mid = (a, b) => a.clone().add(b).multiplyScalar(0.5);
const centroid = pts => pts.reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(1 / pts.length);

// ---------------- analysis per tool ----------------

/** Build the saved result (values + 3D graphics) for a finished tool. */
export function buildResult(type, points, ctx) {
  const { terrain, geo, viewer, options } = ctx;
  const group = new THREE.Group();
  const r = { type, points: points.map(p => p.clone()), group, values: [], primary: null, note: '', visible: true };
  const heightOf = z => geo ? geo.height(z) : z;

  if (type === 'point' || type === 'note') {
    const p = points[0];
    const color = type === 'note' ? COLORS.accent : COLORS.measure;
    group.add(markers([p], color, 13));
    const d = geo?.describe(p);
    r.coords = d;
    // Position in the model's own frame: origin (0,0,0) is the first camera, X east, Y north, Z up (metres).
    const xyz = `${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}`;
    r.primary = type === 'note' ? null : { value: `(${xyz}) m`, sub: 'X, Y, Z from model origin (0,0,0)' };
    r.values = [
      ['X (East)', `${p.x.toFixed(2)} m`],
      ['Y (North)', `${p.y.toFixed(2)} m`],
      ['Z (Up)', `${p.z.toFixed(2)} m`],
    ];
    if (d) r.values.push([geo.heightLabel, fmt.height(d.height)]);
    r.labelObj = label(type === 'note' ? 'Note' : `(${xyz})`, p, type === 'note' ? 'note' : '');
    group.add(r.labelObj);
    return r;
  }

  if (type === 'distance') {
    group.add(fatLine(points, COLORS.measure), markers(points, COLORS.measure));
    const total3d = pathLength(points, false), totalH = pathLength(points, true);
    const dz = points.at(-1).z - points[0].z;
    r.primary = { value: fmt.length(total3d), sub: '3D length' };
    r.values = [
      ['Ground (horizontal) length', fmt.length(totalH)],
      ['Height change start → end', `${dz >= 0 ? '+' : ''}${fmt.height(dz)}`],
      ['Segments', String(points.length - 1)],
    ];
    if (points.length > 2) {
      for (let i = 0; i < points.length - 1; i++) {
        group.add(label(fmt.length(points[i].distanceTo(points[i + 1])), mid(points[i], points[i + 1]), 'dim'));
      }
    }
    r.labelObj = label(fmt.length(total3d), points.at(-1));
    group.add(r.labelObj);
    return r;
  }

  if (type === 'height') {
    const [a, b] = points;
    const low = a.z <= b.z ? a : b, high = a.z <= b.z ? b : a;
    const corner = new THREE.Vector3(low.x, low.y, high.z);
    const vertical = high.z - low.z;
    const horizontal = Math.hypot(high.x - low.x, high.y - low.y);
    group.add(fatLine([low, corner], COLORS.measure), fatLine([corner, high], COLORS.measure, { dashed: true, width: 2 }), markers([low, high], COLORS.measure));
    r.primary = { value: fmt.height(vertical), sub: 'vertical height' };
    r.values = [
      ['Base', fmt.height(heightOf(low.z))],
      ['Top', fmt.height(heightOf(high.z))],
      ['Horizontal offset', fmt.length(horizontal)],
      ['Straight-line distance', fmt.length(a.distanceTo(b))],
    ];
    r.labelObj = label(fmt.height(vertical), mid(low, corner));
    group.add(r.labelObj);
    return r;
  }

  if (type === 'area' || type === 'volume') {
    const ring = [...points, points[0]];
    group.add(fill(points, COLORS.measure), fatLine(ring, COLORS.measure), markers(points, COLORS.measure));
    const c = centroid(points);
    if (type === 'area') {
      const s = terrain.areaStats(points);
      r.primary = { value: fmt.area(s.planar), sub: 'plan area' };
      r.values = [
        ['Surface (3D) area', fmt.area(s.surface)],
        ['Perimeter', fmt.length(s.perimeter)],
        ['Lowest / highest', `${fmt.height(heightOf(s.zMin))} / ${fmt.height(heightOf(s.zMax))}`],
        ['Height range', fmt.height(s.zMax - s.zMin)],
      ];
      r.coverage = s.coverage;
      r.labelObj = label(fmt.area(s.planar), new THREE.Vector3(c.x, c.y, Number.isFinite(s.zMax) ? s.zMax : c.z));
    } else {
      const v = terrain.volume(points, options.baseMode);
      const fillDominant = v.fill >= v.cut;
      r.primary = { value: fmt.volume(fillDominant ? v.fill : v.cut), sub: fillDominant ? 'above base (fill)' : 'below base (cut)' };
      r.values = [
        ['Fill (above base)', fmt.volume(v.fill)],
        ['Cut (below base)', fmt.volume(v.cut)],
        ['Net', `${v.net >= 0 ? '+' : '−'}${fmt.volume(Math.abs(v.net))}`],
        ['Footprint', fmt.area(v.area)],
        ['Base', options.baseMode === 'lowest' ? 'Flat at lowest outline point' : 'Plane fitted to outline'],
      ];
      r.coverage = v.coverage;
      r.labelObj = label(fmt.volume(fillDominant ? v.fill : v.cut), c);
    }
    if (r.coverage < 0.9) {
      r.note = `Only ${fmt.percent(r.coverage, 0)} of this outline has 3D data; empty parts are not counted.`;
    }
    group.add(r.labelObj);
    return r;
  }

  if (type === 'profile') {
    const pr = terrain.profile(points);
    const draped = points.map(p => new THREE.Vector3(p.x, p.y, Number.isFinite(terrain.sample(p.x, p.y)) ? terrain.sample(p.x, p.y) : p.z));
    group.add(fatLine(draped, COLORS.accent), markers(draped, COLORS.accent));
    r.profile = pr;
    r.primary = { value: fmt.length(pr.length), sub: 'route length' };
    r.values = [
      ['Along the surface', fmt.length(pr.surfaceLength)],
      ['Lowest / highest', `${fmt.height(heightOf(pr.zMin))} / ${fmt.height(heightOf(pr.zMax))}`],
      ['Total climb / descent', `+${fmt.height(pr.gain)} / −${fmt.height(pr.loss)}`],
      ['Average grade (start → end)', fmt.slope(pr.avgGrade)],
      [`Steepest over ${Math.round(pr.slopeWindow)} m`, fmt.slope(pr.maxSlope)],
    ];
    r.coverage = pr.coverage;
    r.note = 'Follows the top surface, including roofs and trees.';
    if (pr.coverage < 0.9) r.note += ` ${fmt.percent(1 - pr.coverage, 0)} of the route has no 3D data (gaps in the chart).`;
    r.labelObj = label(fmt.length(pr.length), draped.at(-1));
    group.add(r.labelObj);
    return r;
  }

  if (type === 'los') {
    const [a, b] = points;
    const eye = a.clone().add(new THREE.Vector3(0, 0, options.observerHeight));
    const tgt = b.clone().add(new THREE.Vector3(0, 0, options.targetHeight));
    let blockPoint = null;
    const hit = viewer.firstObstruction(eye, tgt);
    if (hit) blockPoint = hit.point.clone();
    else if (!viewer.mesh) {
      const res = terrain.lineOfSight(eye, tgt);
      if (!res.visible) blockPoint = new THREE.Vector3(res.point.x, res.point.y, res.point.z);
    }
    const visible = !blockPoint;
    group.add(fatLine([a, eye], COLORS.white, { width: 2 }), fatLine([b, tgt], COLORS.white, { width: 2 }));
    if (visible) group.add(fatLine([eye, tgt], COLORS.good, { width: 4 }));
    else group.add(fatLine([eye, blockPoint], COLORS.good, { width: 4 }), fatLine([blockPoint, tgt], COLORS.bad, { width: 4, dashed: true }), markers([blockPoint], COLORS.bad, 14));
    group.add(markers([eye, tgt], COLORS.white));
    const dist = eye.distanceTo(tgt);
    const angle = THREE.MathUtils.radToDeg(Math.atan2(tgt.z - eye.z, Math.hypot(tgt.x - eye.x, tgt.y - eye.y)));
    r.visible3d = visible;
    r.primary = { value: visible ? 'Visible' : 'Blocked', sub: visible ? 'clear line of sight' : `at ${fmt.length(eye.distanceTo(blockPoint))} from observer`, verdict: visible ? 'good' : 'bad' };
    r.values = [
      ['Distance', fmt.length(dist)],
      ['Observer / target above surface', `${fmt.height(options.observerHeight)} / ${fmt.height(options.targetHeight)}`],
      ['Look angle', `${angle >= 0 ? '+' : ''}${angle.toFixed(1)}°`],
    ];
    if (!visible) r.values.splice(1, 0, ['Obstruction height', fmt.height(heightOf(blockPoint.z))]);
    r.note = 'Tested against the 3D surface; objects within 2 m of either position are ignored.';
    r.labelObj = label(visible ? 'Visible' : 'Blocked', visible ? mid(eye, tgt) : blockPoint, visible ? 'good' : 'bad');
    group.add(r.labelObj);
    return r;
  }
  throw new Error(`Unknown tool ${type}`);
}

/** Lightweight live readout while points are being placed. */
export function liveReadout(type, points, cursor) {
  const pts = cursor ? [...points, cursor] : points;
  if (type === 'distance' || type === 'profile') {
    if (pts.length < 2) return null;
    return { value: fmt.length(pathLength(pts, type === 'distance' ? false : true)), sub: `${points.length} point${points.length === 1 ? '' : 's'} placed` };
  }
  if (type === 'area' || type === 'volume') {
    if (pts.length < 3) return { value: '—', sub: `${points.length} of at least 3 corners` };
    return { value: fmt.area(polygonArea(pts)), sub: `${points.length} corners placed` };
  }
  if (type === 'height' && pts.length === 2) {
    return { value: fmt.height(Math.abs(pts[1].z - pts[0].z)), sub: 'vertical height' };
  }
  if (type === 'los' && pts.length === 2) {
    return { value: fmt.length(pts[0].distanceTo(pts[1])), sub: 'observer to target' };
  }
  return null;
}

/** Temporary graphics while a tool is in progress. */
export function previewGroup(type, points, cursor) {
  const group = new THREE.Group();
  const pts = cursor ? [...points, cursor] : [...points];
  if (points.length) group.add(markers(points, COLORS.measure));
  if (pts.length >= 2) {
    const closed = (type === 'area' || type === 'volume') && pts.length >= 3 ? [...pts, pts[0]] : pts;
    group.add(fatLine(closed, COLORS.measure, { width: 2.5, opacity: 0.9 }));
  }
  if (cursor) group.add(markers([cursor], COLORS.white, 9));
  return group;
}
