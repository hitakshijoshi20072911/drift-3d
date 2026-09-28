// Local ENU (metres, x = East, y = North, z = Up) <-> WGS84, UTM and MGRS.
import proj4 from 'proj4';
import { forward as mgrsForward } from 'mgrs';

const A = 6378137.0;
const F = 1 / 298.257223563;
const E2 = F * (2 - F);
const RAD = Math.PI / 180;

function geodeticToEcef(latDeg, lonDeg, h) {
  const lat = latDeg * RAD, lon = lonDeg * RAD;
  const s = Math.sin(lat), c = Math.cos(lat);
  const n = A / Math.sqrt(1 - E2 * s * s);
  return [(n + h) * c * Math.cos(lon), (n + h) * c * Math.sin(lon), (n * (1 - E2) + h) * s];
}

function ecefToGeodetic(x, y, z) {
  const lon = Math.atan2(y, x);
  const p = Math.hypot(x, y);
  let lat = Math.atan2(z, p * (1 - E2));
  let h = 0;
  for (let i = 0; i < 6; i++) {
    const s = Math.sin(lat);
    const n = A / Math.sqrt(1 - E2 * s * s);
    h = p / Math.cos(lat) - n;
    lat = Math.atan2(z, p * (1 - E2 * n / (n + h)));
  }
  return { lat: lat / RAD, lon: lon / RAD, h };
}

export class GeoFrame {
  /**
   * @param {{latitude:number, longitude:number, altitude_m?:number}} origin ENU origin
   * @param {object|null} verticalReference georeference.json vertical_reference, if any
   */
  constructor(origin, verticalReference = null) {
    this.origin = origin;
    this.vertical = verticalReference;
    const lat0 = origin.latitude * RAD, lon0 = origin.longitude * RAD;
    this.sl = Math.sin(lat0); this.cl = Math.cos(lat0);
    this.so = Math.sin(lon0); this.co = Math.cos(lon0);
    this.ecef0 = geodeticToEcef(origin.latitude, origin.longitude, origin.altitude_m || 0);
    const zone = Math.floor((origin.longitude + 180) / 6) + 1;
    this.utmZone = zone;
    this.utmSouth = origin.latitude < 0;
    this.utmDef = `+proj=utm +zone=${zone}${this.utmSouth ? ' +south' : ''} +datum=WGS84 +units=m +no_defs`;
  }

  /** Height label that states the vertical datum honestly. */
  get heightLabel() {
    if (this.aboveGround) return 'Height above ground';
    const v = this.vertical;
    if (v && v.heights_are_absolute) return `Elevation${v.datums?.length ? ` (${v.datums.join(', ')})` : ''}`;
    if (v && v.heights_are_absolute === false) return 'Height (telemetry datum)';
    return 'Height';
  }

  toGeodetic(e, n, u) {
    const { sl, cl, so, co } = this;
    const dx = -so * e - sl * co * n + cl * co * u;
    const dy = co * e - sl * so * n + cl * so * u;
    const dz = cl * n + sl * u;
    return ecefToGeodetic(this.ecef0[0] + dx, this.ecef0[1] + dy, this.ecef0[2] + dz);
  }

  /**
   * Set the model's ground level (local up value). When the flight has no usable
   * origin altitude, heights are then reported above that ground instead of as
   * large negative offsets below the first camera.
   */
  setGround(groundZ) {
    this.groundZ = groundZ;
    const alt = Number(this.origin.altitude_m);
    this.aboveGround = !Number.isFinite(alt) || alt === 0 || alt + groundZ < 0;
  }

  /** Height of an ENU point: elevation in the telemetry frame, or height above ground. */
  height(u) {
    if (this.aboveGround) return u - this.groundZ;
    return (this.origin.altitude_m || 0) + u;
  }

  describe(point) {
    const g = this.toGeodetic(point.x, point.y, point.z);
    const [east, north] = proj4('WGS84', this.utmDef, [g.lon, g.lat]);
    let mgrs = '';
    try {
      const raw = mgrsForward([g.lon, g.lat], 5);
      const m = raw.match(/^(\d{1,2}[C-X])([A-Z]{2})(\d+)$/);
      mgrs = m ? `${m[1]} ${m[2]} ${m[3].slice(0, m[3].length / 2)} ${m[3].slice(m[3].length / 2)}` : raw;
    } catch { mgrs = '—'; }
    return {
      lat: g.lat,
      lon: g.lon,
      height: this.height(point.z),
      latLon: `${g.lat.toFixed(6)}°, ${g.lon.toFixed(6)}°`,
      utm: `${this.utmZone}${this.utmSouth ? 'S' : 'N'} ${east.toFixed(1)} E ${north.toFixed(1)} N`,
      mgrs,
    };
  }
}

export const fmt = {
  length(m) {
    if (!Number.isFinite(m)) return '—';
    const a = Math.abs(m);
    if (a >= 1000) return `${(m / 1000).toFixed(a >= 10000 ? 1 : 2)} km`;
    if (a >= 100) return `${m.toFixed(1)} m`;
    return `${m.toFixed(2)} m`;
  },
  area(m2) {
    if (!Number.isFinite(m2)) return '—';
    if (m2 >= 1e6) return `${(m2 / 1e6).toFixed(3)} km²`;
    if (m2 >= 1e5) return `${(m2 / 1e4).toFixed(2)} ha`;
    return m2 >= 100 ? `${Math.round(m2).toLocaleString()} m²` : `${m2.toFixed(1)} m²`;
  },
  volume(m3) {
    if (!Number.isFinite(m3)) return '—';
    if (m3 >= 1e6) return `${(m3 / 1e6).toFixed(2)} million m³`;
    return `${Math.round(m3).toLocaleString()} m³`;
  },
  height(m) {
    return Number.isFinite(m) ? `${m.toFixed(1)} m` : '—';
  },
  percent(x, digits = 1) {
    return Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : '—';
  },
  slope(ratio) {
    if (!Number.isFinite(ratio)) return '—';
    return `${(ratio * 100).toFixed(1)}% (${(Math.atan(ratio) / RAD).toFixed(1)}°)`;
  },
  duration(seconds) {
    if (!Number.isFinite(seconds)) return '—';
    const total = Math.round(seconds);
    const m = Math.floor(total / 60), s = total % 60;
    return `${m}:${String(s).padStart(2, '0')} min`;
  },
};
