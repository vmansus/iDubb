import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3005,
    host: '0.0.0.0',  // Allow external access for Docker
    proxy: {
      '/api': {
        // In Docker: backend:8888, locally: localhost:8888
        target: process.env.VITE_API_URL || 'http://localhost:8888',
        changeOrigin: true,
        secure: false,
        // Rewrite is not needed since paths match
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Proxying:', req.method, req.url, '-> target');
          });
        },
      },
    },
  },
})
