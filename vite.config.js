import { defineConfig } from 'vite'
import { cpSync } from 'node:fs'

// COOP/COEP 开启 Cross-Origin Isolation：
// 1) 允许 onnxruntime-web 使用 SharedArrayBuffer 多线程 WASM（兜底加速）
// 2) WebGPU 本身不需要，但需要安全上下文（localhost/HTTPS 天然满足）
export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
  build: {
    // ORT 运行时文件（ort/）在 dev 下由 Vite 直接服务，
    // 构建产物需手动复制（动态 import 的路径运行时拼接，无法静态打包）
    rollupOptions: {
      plugins: [
        {
          name: 'copy-ort-runtime',
          closeBundle() {
            cpSync('ort', 'dist/ort', { recursive: true })
          },
        },
      ],
    },
  },
})
