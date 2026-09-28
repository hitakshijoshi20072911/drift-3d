import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths: the same build works at a domain root, under a
  // GitHub Pages project path (/drone-video-to-3d/) or opened offline.
  base: './',
  build: {
    // One offline bundle (three.js + analysis code) is intentional.
    chunkSizeWarningLimit: 1200,
  },
});
