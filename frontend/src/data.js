import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

const JSON_FILES = ['viewer_metadata.json', 'run_report.json', 'verification_report.json', 'georeference.json', 'gpu_usage.summary.json', 'confidence_summary.json'];
const MODEL_FILES = ['model.glb', 'mesh.ply'];
const CLOUD_FILES = ['point_cloud.ply'];

/** Pick the files the viewer understands from a flat list of File objects. */
export function classifyFiles(files) {
  const byName = new Map();
  let folder = '';
  for (const f of files) {
    const rel = f.webkitRelativePath || f.relativePath || f.name;
    const parts = rel.split('/');
    if (!folder && parts.length > 1) folder = parts[0];
    // prefer top-level files over nested copies (e.g. textured/mesh.ply)
    const depth = parts.length;
    const key = f.name.toLowerCase();
    if (!byName.has(key) || depth < byName.get(key).depth) byName.set(key, { file: f, depth });
  }
  const get = name => byName.get(name)?.file ?? null;
  return {
    folder,
    model: MODEL_FILES.map(get).find(Boolean) ?? [...byName.values()].map(v => v.file).find(f => /\.glb$/i.test(f.name)) ?? null,
    cloud: CLOUD_FILES.map(get).find(Boolean) ?? null,
    json: Object.fromEntries(JSON_FILES.map(n => [n, get(n)]).filter(([, f]) => f)),
  };
}

/** Recursively read a dropped folder (DataTransferItem entries). */
export async function filesFromDrop(dataTransfer) {
  const items = [...(dataTransfer.items || [])].map(i => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!items.length) return [...dataTransfer.files];
  const out = [];
  const walk = async (entry, path, depth) => {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      file.relativePath = path + file.name;
      out.push(file);
    } else if (entry.isDirectory && depth < 3) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walk(child, `${path}${entry.name}/`, depth + 1);
      } while (batch.length);
    }
  };
  for (const entry of items) await walk(entry, '', 0);
  return out;
}

function readWithProgress(file, onProgress) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onprogress = e => e.lengthComputable && onProgress(e.loaded / e.total);
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded / total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out.buffer;
}

function parseGlb(buffer) {
  return new Promise((resolve, reject) => new GLTFLoader().parse(buffer, '', resolve, reject));
}

/** Merge every mesh of a glTF scene into one surface mesh with an unlit photo material. */
function surfaceFromGltf(gltf) {
  gltf.scene.updateMatrixWorld(true);
  const meshes = [];
  gltf.scene.traverse(o => { if (o.isMesh) meshes.push(o); });
  if (!meshes.length) return null;
  const src = meshes[0];
  const geometry = src.geometry.clone().applyMatrix4(src.matrixWorld);
  const mat = Array.isArray(src.material) ? src.material[0] : src.material;
  return buildSurface(geometry, mat?.map ?? null, meshes.length);
}

function buildSurface(geometry, map, parts = 1) {
  let photoMaterial;
  if (map) {
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = 8;
    photoMaterial = new THREE.MeshBasicMaterial({ map, side: THREE.DoubleSide });
  } else if (geometry.attributes.color) {
    photoMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  } else {
    geometry.computeVertexNormals();
    photoMaterial = new THREE.MeshLambertMaterial({ color: '#b9c6d0', side: THREE.DoubleSide });
  }
  if (!geometry.attributes.normal) geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, photoMaterial);
  mesh.userData.photoMaterial = photoMaterial;
  mesh.userData.textured = Boolean(map);
  mesh.userData.parts = parts;
  return mesh;
}

function cloudFromPly(geometry) {
  const material = new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: Boolean(geometry.attributes.color) });
  if (!geometry.attributes.color) material.color.set('#c9d4de');
  return new THREE.Points(geometry, material);
}

/**
 * Load a reconstruction from local files or URLs.
 * @param {{model?:File|string, cloud?:File|string, json:Record<string, File|string>}} source
 * @param {(stage:string, fraction:number)=>void} progress
 */
export async function loadReconstruction(source, progress) {
  const read = (item, label, weight0, weight1) => (typeof item === 'string'
    ? fetchWithProgress(item, f => progress(label, weight0 + f * (weight1 - weight0)))
    : readWithProgress(item, f => progress(label, weight0 + f * (weight1 - weight0))));

  const meta = {};
  for (const [name, item] of Object.entries(source.json || {})) {
    try {
      const text = typeof item === 'string' ? await (await fetch(item)).text() : await item.text();
      meta[name.replace('.json', '')] = JSON.parse(text);
    } catch { /* optional file */ }
  }

  let mesh = null, cloud = null;
  if (source.model) {
    const name = typeof source.model === 'string' ? source.model : source.model.name;
    const buffer = await read(source.model, `Reading ${name.split('/').pop()}`, 0, source.cloud ? 0.6 : 0.9);
    progress('Preparing the 3D model', source.cloud ? 0.62 : 0.92);
    if (/\.glb$/i.test(name)) mesh = surfaceFromGltf(await parseGlb(buffer));
    else {
      const geometry = new PLYLoader().parse(buffer);
      mesh = geometry.index ? buildSurface(geometry, null) : null;
      if (!mesh) cloud = cloudFromPly(geometry);
    }
  }
  if (source.cloud && !cloud) {
    const buffer = await read(source.cloud, 'Reading point_cloud.ply', source.model ? 0.62 : 0, 0.92);
    progress('Preparing the point cloud', 0.94);
    cloud = cloudFromPly(new PLYLoader().parse(buffer));
  }
  return { mesh, cloud, meta };
}

/** First value stored under `key` anywhere in a nested object. */
export function findKey(obj, key) {
  if (!obj || typeof obj !== 'object') return undefined;
  if (Object.prototype.hasOwnProperty.call(obj, key)) return obj[key];
  for (const v of Object.values(obj)) {
    const found = findKey(v, key);
    if (found !== undefined) return found;
  }
  return undefined;
}
