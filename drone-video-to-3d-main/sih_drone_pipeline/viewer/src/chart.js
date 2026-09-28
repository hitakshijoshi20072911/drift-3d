import { fmt } from './geo.js';

/** Elevation profile chart; empty (NaN) stretches are drawn as gaps, never interpolated. */
export class ProfileChart {
  constructor(canvas, onHover) {
    this.canvas = canvas;
    this.onHover = onHover;
    this.profile = null;
    this.hover = -1;
    canvas.addEventListener('pointermove', e => this._move(e));
    canvas.addEventListener('pointerleave', () => { this.hover = -1; this.draw(); this.onHover?.(-1); });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  set(profile, heightOf = z => z) {
    this.profile = profile;
    this.heights = profile.z.map(z => (Number.isNaN(z) ? NaN : heightOf(z)));
    this.hover = -1;
    this.draw();
  }

  _geom() {
    const dpr = devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const pad = { l: 58, r: 14, t: 10, b: 24 };
    return { dpr, w, h, pad, pw: w - pad.l - pad.r, ph: h - pad.t - pad.b };
  }

  _move(e) {
    if (!this.profile) return;
    const { pad, pw } = this._geom();
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - pad.l;
    const d = (x / pw) * this.profile.length;
    let best = -1, bestDist = Infinity;
    this.profile.d.forEach((v, i) => {
      const dd = Math.abs(v - d);
      if (dd < bestDist && !Number.isNaN(this.heights[i])) { bestDist = dd; best = i; }
    });
    this.hover = best;
    this.draw();
    this.onHover?.(best);
  }

  draw() {
    const g = this.canvas.getContext('2d');
    const { dpr, w, h, pad, pw, ph } = this._geom();
    if (!w || !h) return;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!this.profile) return;
    const { d, length } = this.profile;
    const zs = this.heights.filter(z => !Number.isNaN(z));
    if (!zs.length) {
      g.fillStyle = '#8a9aa9';
      g.fillText('No 3D data along this route', pad.l, pad.t + 20);
      return;
    }
    let zmin = Math.min(...zs), zmax = Math.max(...zs);
    const span = Math.max(zmax - zmin, 1);
    zmin -= span * 0.08; zmax += span * 0.12;
    const X = v => pad.l + (v / (length || 1)) * pw;
    const Y = z => pad.t + (1 - (z - zmin) / (zmax - zmin)) * ph;
    const css = getComputedStyle(document.documentElement);
    const muted = css.getPropertyValue('--muted').trim() || '#8a9aa9';
    const grid = css.getPropertyValue('--border').trim() || '#243240';

    g.font = '11px ' + (css.getPropertyValue('--font') || 'sans-serif');
    g.textBaseline = 'middle';
    g.strokeStyle = grid; g.fillStyle = muted; g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const z = zmin + ((zmax - zmin) * i) / 4;
      const y = Y(z);
      g.beginPath(); g.moveTo(pad.l, y); g.lineTo(pad.l + pw, y); g.stroke();
      g.textAlign = 'right'; g.fillText(`${z.toFixed(0)} m`, pad.l - 8, y);
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let i = 0; i <= 5; i++) {
      const v = (length * i) / 5;
      g.fillText(fmt.length(v), X(v), pad.t + ph + 6);
    }

    // filled area and line, broken at gaps
    const segments = [];
    let cur = [];
    this.heights.forEach((z, i) => {
      if (Number.isNaN(z)) { if (cur.length) segments.push(cur); cur = []; } else cur.push(i);
    });
    if (cur.length) segments.push(cur);
    const grad = g.createLinearGradient(0, pad.t, 0, pad.t + ph);
    grad.addColorStop(0, '#4fa3ff55'); grad.addColorStop(1, '#4fa3ff05');
    for (const seg of segments) {
      g.beginPath();
      g.moveTo(X(d[seg[0]]), pad.t + ph);
      seg.forEach(i => g.lineTo(X(d[i]), Y(this.heights[i])));
      g.lineTo(X(d[seg.at(-1)]), pad.t + ph);
      g.closePath(); g.fillStyle = grad; g.fill();
      g.beginPath();
      seg.forEach((i, k) => (k ? g.lineTo : g.moveTo).call(g, X(d[i]), Y(this.heights[i])));
      g.strokeStyle = '#4fa3ff'; g.lineWidth = 2; g.stroke();
    }

    if (this.hover >= 0) {
      const i = this.hover, x = X(d[i]), y = Y(this.heights[i]);
      g.strokeStyle = '#e6edf3'; g.lineWidth = 1; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, pad.t + ph); g.stroke(); g.setLineDash([]);
      g.beginPath(); g.arc(x, y, 4.5, 0, Math.PI * 2); g.fillStyle = '#ffb224'; g.fill();
      const text = `${fmt.length(d[i])}  ·  ${this.heights[i].toFixed(1)} m`;
      g.font = '600 12px ' + (css.getPropertyValue('--font') || 'sans-serif');
      const tw = g.measureText(text).width + 14;
      const tx = Math.min(Math.max(x - tw / 2, pad.l), pad.l + pw - tw);
      g.fillStyle = '#0b1015ee'; g.strokeStyle = '#ffb224';
      g.beginPath(); g.roundRect(tx, pad.t + 2, tw, 22, 5); g.fill(); g.stroke();
      g.fillStyle = '#fff'; g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillText(text, tx + 7, pad.t + 13);
    }
  }
}
