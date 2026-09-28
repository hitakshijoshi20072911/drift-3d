import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { computeBoundsTree, acceleratedRaycast, PointsBVH } from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;
THREE.Points.prototype.raycast = acceleratedRaycast;
THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // ENU: z is up

// Perceptually ordered height ramp (low -> high).
const RAMP = ['#2c1a6b', '#1f5fa8', '#1a9e9a', '#5cc35a', '#e8d33c', '#f28a2e', '#d8412f'];
export const RAMP_CSS = `linear-gradient(90deg, ${RAMP.join(', ')})`;
const rampColors = RAMP.map(c => new THREE.Color(c));

function rampAt(t, out) {
  const x = Math.min(0.9999, Math.max(0, t)) * (rampColors.length - 1);
  const i = Math.floor(x);
  return out.copy(rampColors[i]).lerp(rampColors[i + 1], x - i);
}

export class Viewer {
  constructor(host) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'label-layer';
    host.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0b1015');
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
    this.camera.position.set(200, -200, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.maxPolarAngle = Math.PI * 0.495;

    this.scene.add(new THREE.HemisphereLight('#dfefff', '#3a3226', 1.6));
    const sun = new THREE.DirectionalLight('#ffffff', 1.8);
    sun.position.set(-0.5, -0.8, 1.2);
    this.scene.add(sun);

    this.modelRoot = new THREE.Group();
    this.overlay = new THREE.Group(); // measurements, always drawn on top
    this.scene.add(this.modelRoot, this.overlay);

    this.mesh = null;       // THREE.Mesh (surface)
    this.cloud = null;      // THREE.Points
    this.display = 'photo';
    this.heightRange = [0, 1];
    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = true;
    this.pointer = new THREE.Vector2();
    this.onFrame = [];

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  resize() {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const cb of this.onResize ?? []) cb(w, h);
  }

  frame() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    for (const cb of this.onFrame) cb();
  }

  /** Register the loaded surface mesh and/or point cloud. */
  setModel({ mesh, cloud }) {
    this.modelRoot.clear();
    this.mesh = mesh || null;
    this.cloud = cloud || null;
    if (mesh) {
      mesh.geometry.computeBoundsTree();
      this.modelRoot.add(mesh);
    }
    if (cloud) {
      cloud.geometry.computeBoundsTree({ type: PointsBVH });
      this.modelRoot.add(cloud);
    }
    const box = new THREE.Box3().setFromObject(this.modelRoot);
    this.box = box;
    const size = box.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y, size.z, 1);
    this.camera.near = Math.max(0.05, span / 20000);
    this.camera.far = span * 50;
    this.camera.updateProjectionMatrix();
    this.controls.maxDistance = span * 8;
    this.raycaster.params.Points.threshold = Math.max(0.05, span / 3000);
  }

  setHeightRange(lo, hi) {
    this.heightRange = [lo, hi];
    const color = new THREE.Color();
    const paint = geometry => {
      const pos = geometry.attributes.position;
      const arr = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        rampAt((pos.getZ(i) - lo) / (hi - lo || 1), color);
        arr[i * 3] = color.r; arr[i * 3 + 1] = color.g; arr[i * 3 + 2] = color.b;
      }
      return new THREE.BufferAttribute(arr, 3);
    };
    if (this.mesh) {
      this.mesh.geometry.setAttribute('heightColor', paint(this.mesh.geometry));
    }
    if (this.cloud) {
      this.cloud.userData.heightColor = paint(this.cloud.geometry);
    }
  }

  /** photo | height | points */
  setDisplay(mode) {
    this.display = mode;
    const { mesh, cloud } = this;
    if (mesh) {
      const ud = mesh.userData;
      mesh.visible = mode !== 'points' || !cloud;
      if (mode === 'height') {
        if (!ud.heightMaterial) {
          ud.heightMaterial = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
        }
        const g = mesh.geometry;
        if (!ud.photoColor && g.attributes.color) ud.photoColor = g.attributes.color;
        g.setAttribute('color', g.attributes.heightColor);
        mesh.material = ud.heightMaterial;
      } else {
        if (ud.photoColor) mesh.geometry.setAttribute('color', ud.photoColor);
        else mesh.geometry.deleteAttribute('color');
        mesh.material = ud.photoMaterial;
      }
    }
    if (cloud) {
      cloud.visible = mode === 'points' || !mesh;
      const ud = cloud.userData;
      if (!ud.photoColor) ud.photoColor = cloud.geometry.attributes.color;
      const useHeight = mode === 'height' || !ud.photoColor;
      if (useHeight && ud.heightColor) cloud.geometry.setAttribute('color', ud.heightColor);
      else if (ud.photoColor) cloud.geometry.setAttribute('color', ud.photoColor);
      cloud.material.vertexColors = true;
      cloud.material.needsUpdate = true;
    }
  }

  // ---------------- camera ----------------

  frameBox(box, { top = false, padding = 1.15, animate = true } = {}) {
    if (!box || box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(Math.hypot(size.x, size.y) / 2, size.z / 2, 2); // frame the ground footprint
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const aspectFit = Math.min(1, this.camera.aspect);
    const dist = (radius * padding) / Math.tan(fov / 2) / Math.max(aspectFit, 0.55);
    let dir;
    if (top) dir = new THREE.Vector3(0, -0.0015, 1);
    else {
      const current = this.camera.position.clone().sub(this.controls.target);
      const az = Math.atan2(current.x, -current.y);
      const useAz = Number.isFinite(az) && current.lengthSq() > 0 ? az : -0.6;
      dir = new THREE.Vector3(Math.sin(useAz) * 0.75, -Math.cos(useAz) * 0.75, 0.85);
    }
    dir.normalize();
    this._animateTo(center, center.clone().add(dir.multiplyScalar(dist)), animate);
  }

  _animateTo(target, position, animate) {
    const fromT = this.controls.target.clone(), fromP = this.camera.position.clone();
    if (!animate || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.controls.target.copy(target); this.camera.position.copy(position); return;
    }
    const start = performance.now(), dur = 450;
    const step = () => {
      const t = Math.min(1, (performance.now() - start) / dur);
      const e = 1 - Math.pow(1 - t, 3);
      this.controls.target.lerpVectors(fromT, target, e);
      this.camera.position.lerpVectors(fromP, position, e);
      if (t < 1) requestAnimationFrame(step);
    };
    step();
  }

  fit() { this.frameBox(this.box); }
  topView() { this.frameBox(this.box, { top: true, padding: 1.0 }); }

  /** Camera heading in degrees (0 = looking north). */
  heading() {
    const d = this.controls.target.clone().sub(this.camera.position);
    return THREE.MathUtils.radToDeg(Math.atan2(d.x, d.y));
  }

  /** Metres per screen pixel at the orbit target. */
  metresPerPixel() {
    const dist = this.camera.position.distanceTo(this.controls.target);
    const h = 2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    return h / (this.renderer.domElement.clientHeight || 1);
  }

  // ---------------- picking ----------------

  pick(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const targets = [];
    if (this.mesh) targets.push(this.mesh);        // the surface is picked even when hidden
    else if (this.cloud) targets.push(this.cloud);
    const hits = this.raycaster.intersectObjects(targets, false);
    if (hits.length) return hits[0].point.clone();
    if (this.terrain) {
      const p = this.terrain.raycast(this.raycaster.ray.origin, this.raycaster.ray.direction);
      if (p) return new THREE.Vector3(p.x, p.y, p.z);
    }
    return null;
  }

  /** First surface hit strictly between two points (true 3D sight-line test, facades included). */
  firstObstruction(a, b, endTolerance = 2) {
    if (!this.mesh) return null;
    const dir = b.clone().sub(a);
    const len = dir.length();
    // Ignore surface noise (grass, small bushes) right at the two positions.
    const clearance = Math.min(endTolerance, len * 0.05);
    dir.normalize();
    const rc = new THREE.Raycaster(a.clone().addScaledVector(dir, clearance), dir, 0, Math.max(0, len - 2 * clearance));
    rc.firstHitOnly = true;
    const hits = rc.intersectObject(this.mesh, false);
    return hits.length ? hits[0] : null;
  }

  project(v) {
    const p = v.clone().project(this.camera);
    const el = this.renderer.domElement;
    return { x: (p.x + 1) / 2 * el.clientWidth, y: (1 - p.y) / 2 * el.clientHeight, visible: p.z < 1 };
  }
}
