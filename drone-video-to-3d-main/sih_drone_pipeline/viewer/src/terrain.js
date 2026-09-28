// 2.5D height grid (a DSM in the local ENU frame) used for area, volume,
// profile and line-of-sight analysis. Cells never observed stay NaN so every
// result can report how much of its region actually had data.

const NICE_RES = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20];

export class HeightField {
  constructor(minX, minY, res, cols, rows) {
    Object.assign(this, { minX, minY, res, cols, rows });
    this.z = new Float32Array(cols * rows).fill(NaN);
  }

  static forBox(box, targetCells = 4e6) {
    const w = Math.max(box.max.x - box.min.x, 1);
    const h = Math.max(box.max.y - box.min.y, 1);
    const ideal = Math.sqrt((w * h) / targetCells);
    const res = NICE_RES.find(r => r >= ideal) ?? Math.ceil(ideal);
    const cols = Math.ceil(w / res) + 1;
    const rows = Math.ceil(h / res) + 1;
    return new HeightField(box.min.x, box.min.y, res, cols, rows);
  }

  /** Rasterise triangles (max height per cell) plus their vertices. */
  addTriangles(pos, index) {
    const { minX, minY, res, cols, rows, z } = this;
    const count = index ? index.length : pos.length / 3;
    for (let i = 0; i < pos.length; i += 3) this._splat(pos[i], pos[i + 1], pos[i + 2]);
    for (let t = 0; t < count; t += 3) {
      const a = (index ? index[t] : t) * 3, b = (index ? index[t + 1] : t + 1) * 3, c = (index ? index[t + 2] : t + 2) * 3;
      const x0 = pos[a], y0 = pos[a + 1], z0 = pos[a + 2];
      const x1 = pos[b], y1 = pos[b + 1], z1 = pos[b + 2];
      const x2 = pos[c], y2 = pos[c + 1], z2 = pos[c + 2];
      const area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
      if (Math.abs(area2) < 1e-9) continue;
      const c0 = Math.max(0, Math.floor((Math.min(x0, x1, x2) - minX) / res));
      const c1 = Math.min(cols - 1, Math.floor((Math.max(x0, x1, x2) - minX) / res));
      const r0 = Math.max(0, Math.floor((Math.min(y0, y1, y2) - minY) / res));
      const r1 = Math.min(rows - 1, Math.floor((Math.max(y0, y1, y2) - minY) / res));
      if ((c1 - c0 + 1) * (r1 - r0 + 1) > 250000) continue; // degenerate bridging face
      for (let r = r0; r <= r1; r++) {
        const cy = minY + (r + 0.5) * res;
        for (let col = c0; col <= c1; col++) {
          const cx = minX + (col + 0.5) * res;
          const w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) / area2;
          const w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) / area2;
          const w2 = 1 - w0 - w1;
          if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
          const h = w0 * z0 + w1 * z1 + w2 * z2;
          const idx = r * cols + col;
          if (!(h <= z[idx])) z[idx] = h;
        }
      }
    }
  }

  addPoints(pos) {
    for (let i = 0; i < pos.length; i += 3) this._splat(pos[i], pos[i + 1], pos[i + 2]);
  }

  _splat(x, y, h) {
    const col = Math.floor((x - this.minX) / this.res);
    const row = Math.floor((y - this.minY) / this.res);
    if (col < 0 || row < 0 || col >= this.cols || row >= this.rows) return;
    const idx = row * this.cols + col;
    if (!(h <= this.z[idx])) this.z[idx] = h;
  }

  /** Close pin-holes (cells with at least 5 observed neighbours). Larger gaps stay empty. */
  fillPinholes(passes = 2) {
    const { cols, rows } = this;
    for (let p = 0; p < passes; p++) {
      const src = this.z.slice();
      for (let r = 1; r < rows - 1; r++) {
        for (let c = 1; c < cols - 1; c++) {
          const idx = r * cols + c;
          if (!Number.isNaN(src[idx])) continue;
          let sum = 0, n = 0;
          for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
            const v = src[idx + dr * cols + dc];
            if (!Number.isNaN(v)) { sum += v; n++; }
          }
          if (n >= 5) this.z[idx] = sum / n;
        }
      }
    }
  }

  cell(col, row) {
    if (col < 0 || row < 0 || col >= this.cols || row >= this.rows) return NaN;
    return this.z[row * this.cols + col];
  }

  /** Bilinear height at (x, y); NaN when no surrounding cell was observed. */
  sample(x, y) {
    const fx = (x - this.minX) / this.res - 0.5;
    const fy = (y - this.minY) / this.res - 0.5;
    const c = Math.floor(fx), r = Math.floor(fy);
    const tx = fx - c, ty = fy - r;
    const v = [this.cell(c, r), this.cell(c + 1, r), this.cell(c, r + 1), this.cell(c + 1, r + 1)];
    const w = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty];
    let sum = 0, wsum = 0;
    for (let i = 0; i < 4; i++) if (!Number.isNaN(v[i])) { sum += v[i] * w[i]; wsum += w[i]; }
    return wsum > 1e-6 ? sum / wsum : NaN;
  }

  percentiles(lo = 0.02, hi = 0.98) {
    const step = Math.max(1, Math.floor(this.z.length / 200000));
    const vals = [];
    for (let i = 0; i < this.z.length; i += step) if (!Number.isNaN(this.z[i])) vals.push(this.z[i]);
    if (!vals.length) return [0, 1];
    vals.sort((a, b) => a - b);
    return [vals[Math.floor(lo * (vals.length - 1))], vals[Math.floor(hi * (vals.length - 1))]];
  }

  /** First intersection of a ray with the surface (fallback picking when no mesh is loaded). */
  raycast(origin, dir, maxDist = 1e6) {
    const step = this.res * 0.5;
    let prev = null;
    for (let t = 0; t < maxDist; t += step) {
      const x = origin.x + dir.x * t, y = origin.y + dir.y * t, zr = origin.z + dir.z * t;
      const col = (x - this.minX) / this.res, row = (y - this.minY) / this.res;
      if (col < -1 || row < -1 || col > this.cols + 1 || row > this.rows + 1) {
        if (prev !== null) break;
        continue;
      }
      const h = this.sample(x, y);
      if (!Number.isNaN(h) && zr <= h) {
        let lo = prev ?? t - step, hi = t;
        for (let i = 0; i < 10; i++) {
          const mid = (lo + hi) / 2;
          const hm = this.sample(origin.x + dir.x * mid, origin.y + dir.y * mid);
          if (!Number.isNaN(hm) && origin.z + dir.z * mid <= hm) hi = mid; else lo = mid;
        }
        return { x: origin.x + dir.x * hi, y: origin.y + dir.y * hi, z: origin.z + dir.z * hi };
      }
      prev = t;
    }
    return null;
  }

  // ---------------- analysis ----------------

  _forCellsInPolygon(poly, fn) {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const p of poly) { minx = Math.min(minx, p.x); maxx = Math.max(maxx, p.x); miny = Math.min(miny, p.y); maxy = Math.max(maxy, p.y); }
    const c0 = Math.max(0, Math.floor((minx - this.minX) / this.res));
    const c1 = Math.min(this.cols - 1, Math.floor((maxx - this.minX) / this.res));
    const r0 = Math.max(0, Math.floor((miny - this.minY) / this.res));
    const r1 = Math.min(this.rows - 1, Math.floor((maxy - this.minY) / this.res));
    for (let r = r0; r <= r1; r++) {
      const cy = this.minY + (r + 0.5) * this.res;
      for (let c = c0; c <= c1; c++) {
        const cx = this.minX + (c + 0.5) * this.res;
        if (pointInPolygon(cx, cy, poly)) fn(c, r, cx, cy, this.z[r * this.cols + c]);
      }
    }
  }

  _boundarySamples(poly) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      const n = Math.max(1, Math.ceil(len / this.res));
      for (let k = 0; k < n; k++) {
        const t = k / n;
        const x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t;
        const h = this.sample(x, y);
        if (!Number.isNaN(h)) out.push({ x, y, z: h });
      }
    }
    return out;
  }

  areaStats(poly) {
    const res2 = this.res * this.res;
    let total = 0, valid = 0, surface = 0, zmin = Infinity, zmax = -Infinity, zsum = 0;
    this._forCellsInPolygon(poly, (c, r, cx, cy, h) => {
      total++;
      if (Number.isNaN(h)) return;
      valid++;
      zsum += h; zmin = Math.min(zmin, h); zmax = Math.max(zmax, h);
      const gx = gradient(this.cell(c - 1, r), h, this.cell(c + 1, r), this.res);
      const gy = gradient(this.cell(c, r - 1), h, this.cell(c, r + 1), this.res);
      surface += res2 * Math.sqrt(1 + gx * gx + gy * gy);
    });
    const planar = polygonArea(poly);
    return {
      planar,
      surface: valid ? surface * (total / valid) : NaN,
      perimeter: polygonPerimeter(poly),
      zMin: valid ? zmin : NaN,
      zMax: valid ? zmax : NaN,
      zMean: valid ? zsum / valid : NaN,
      coverage: total ? valid / total : 0,
    };
  }

  /** Material above (fill) and below (cut) a base surface under the polygon. */
  volume(poly, baseMode = 'fit') {
    const edge = this._boundarySamples(poly);
    let plane;
    if (baseMode === 'lowest' && edge.length) {
      const zmin = Math.min(...edge.map(p => p.z));
      plane = { a: 0, b: 0, c: zmin };
    } else {
      plane = fitPlane(edge);
    }
    const res2 = this.res * this.res;
    let fill = 0, cut = 0, total = 0, valid = 0;
    this._forCellsInPolygon(poly, (c, r, cx, cy, h) => {
      total++;
      if (Number.isNaN(h)) return;
      valid++;
      const d = h - (plane.a * cx + plane.b * cy + plane.c);
      if (d > 0) fill += d * res2; else cut -= d * res2;
    });
    return {
      fill, cut, net: fill - cut,
      area: polygonArea(poly),
      coverage: total ? valid / total : 0,
      plane,
      edgeSamples: edge.length,
    };
  }

  profile(path) {
    const step = Math.max(this.res, pathLength(path) / 2000);
    const d = [], z = [], xs = [], ys = [];
    let acc = 0;
    for (let i = 0; i < path.length - 1; i++) {
      const a = path[i], b = path[i + 1];
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      const n = Math.max(1, Math.ceil(len / step));
      for (let k = 0; k < n + (i === path.length - 2 ? 1 : 0); k++) {
        const t = k / n;
        const x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t;
        d.push(acc + len * t); xs.push(x); ys.push(y); z.push(this.sample(x, y));
      }
      acc += len;
    }
    let zmin = Infinity, zmax = -Infinity, gain = 0, loss = 0, maxSlope = 0, valid = 0, surface = 0;
    let lastIdx = -1;
    const window = Math.max(5, 4 * this.res); // slope over >= 5 m so single walls do not dominate
    for (let i = 0; i < z.length; i++) {
      if (Number.isNaN(z[i])) continue;
      valid++;
      zmin = Math.min(zmin, z[i]); zmax = Math.max(zmax, z[i]);
      if (lastIdx >= 0) {
        const dz = z[i] - z[lastIdx], dd = d[i] - d[lastIdx];
        if (dz > 0) gain += dz; else loss -= dz;
        surface += Math.hypot(dz, dd);
      }
      lastIdx = i;
    }
    for (let i = 0, j = 0; i < z.length; i++) {
      if (Number.isNaN(z[i])) continue;
      while (j < z.length && (d[j] - d[i] < window || Number.isNaN(z[j]))) j++;
      if (j < z.length) maxSlope = Math.max(maxSlope, Math.abs(z[j] - z[i]) / (d[j] - d[i]));
    }
    const firstIdx = z.findIndex(v => !Number.isNaN(v));
    const avgGrade = valid >= 2 && d[lastIdx] > d[firstIdx] ? (z[lastIdx] - z[firstIdx]) / (d[lastIdx] - d[firstIdx]) : NaN;
    return {
      d, z, x: xs, y: ys,
      length: acc,
      avgGrade, slopeWindow: window,
      surfaceLength: surface,
      zMin: valid ? zmin : NaN,
      zMax: valid ? zmax : NaN,
      gain, loss, maxSlope,
      coverage: z.length ? valid / z.length : 0,
    };
  }

  /** Is the straight line between two points clear of the surface? */
  lineOfSight(a, b, clearance = 0.15) {
    const len = Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
    const step = Math.max(this.res * 0.5, len / 4000);
    const skip = Math.min(2, len * 0.05); // same end tolerance as the mesh test
    let unknown = 0, samples = 0;
    for (let t = skip; t < len - skip; t += step) {
      const f = t / len;
      const x = a.x + (b.x - a.x) * f, y = a.y + (b.y - a.y) * f, zl = a.z + (b.z - a.z) * f;
      const h = this.sample(x, y);
      samples++;
      if (Number.isNaN(h)) { unknown++; continue; }
      if (h > zl + clearance) return { visible: false, distance: t, point: { x, y, z: h }, unknown: unknown / samples };
    }
    return { visible: true, unknown: samples ? unknown / samples : 0 };
  }
}

function gradient(prev, mid, next, res) {
  const hasP = !Number.isNaN(prev), hasN = !Number.isNaN(next);
  if (hasP && hasN) return (next - prev) / (2 * res);
  if (hasN) return (next - mid) / res;
  if (hasP) return (mid - prev) / res;
  return 0;
}

export function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function polygonArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    s += a.x * b.y - b.x * a.y;
  }
  return Math.abs(s) / 2;
}

export function polygonPerimeter(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    s += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return s;
}

export function pathLength(path, horizontal = true) {
  let s = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1];
    s += horizontal ? Math.hypot(b.x - a.x, b.y - a.y) : Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
  }
  return s;
}

/** Least-squares plane z = a·x + b·y + c; flat mean plane when under-determined. */
export function fitPlane(pts) {
  if (!pts.length) return { a: 0, b: 0, c: 0 };
  const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const my = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const mz = pts.reduce((s, p) => s + p.z, 0) / pts.length;
  if (pts.length < 3) return { a: 0, b: 0, c: mz };
  let sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0;
  for (const p of pts) {
    const x = p.x - mx, y = p.y - my, z = p.z - mz;
    sxx += x * x; sxy += x * y; syy += y * y; sxz += x * z; syz += y * z;
  }
  const det = sxx * syy - sxy * sxy;
  if (Math.abs(det) < 1e-9) return { a: 0, b: 0, c: mz };
  const a = (sxz * syy - syz * sxy) / det;
  const b = (syz * sxx - sxz * sxy) / det;
  return { a, b, c: mz - a * mx - b * my };
}
