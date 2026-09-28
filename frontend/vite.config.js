import { cpSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const copyStaticDemos = () => ({
  name: 'copy-static-demos',
  closeBundle() {
    const root = resolve(fileURLToPath(new URL('.', import.meta.url)));
    cpSync(resolve(root, 'demo'), resolve(root, 'dist/demo'), { recursive: true });
    cpSync(resolve(root, 'demo_manifest.json'), resolve(root, 'dist/demo_manifest.json'));
    cpSync(resolve(root, '../data/demo_benchmark_metrics.json'), resolve(root, 'dist/demo_benchmark_metrics.json'));
  },
});

export default defineConfig({
  plugins: [copyStaticDemos()],
  // Relative asset paths: the same build works at a domain root, under a
  // GitHub Pages project path (/drone-video-to-3d/) or opened offline.
  base: './',
  build: {
    // One offline bundle (three.js + analysis code) is intentional.
    chunkSizeWarningLimit: 1200,
  },
  server: {
    allowedHosts: ['localhost', '127.0.0.1', '.manus.computer'],
    proxy: {
      '/api': process.env.DRIFTX_API_PROXY || 'http://127.0.0.1:8123',
    },
  },
});
