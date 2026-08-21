import { defineConfig } from 'vite'
import { cpSync, rmSync, readdirSync, existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'

// 模型分片：165MB 超静态托管 25 MiB 单文件限制，构建时切成 19 MiB/片
// （留足余量），运行时同源下载拼接（见 src/main.js loadModelParts）
const MODEL_FILE = 'public/models/htdemucs_fp16weights.onnx'
const PART_SIZE = 19 * 1024 * 1024

/** 通用切片：读源文件 → 19 MiB/片 + manifest 写入 outDir/子目录，删除原文件 */
function sliceFile(src, outDir, name) {
  if (!existsSync(src)) return
  const data = readFileSync(src)
  const n = Math.ceil(data.length / PART_SIZE)
  mkdirSync(outDir, { recursive: true })
  for (let i = 0; i < n; i++) {
    const part = data.subarray(i * PART_SIZE, Math.min((i + 1) * PART_SIZE, data.length))
    writeFileSync(`${outDir}/${name}.part${i}`, part)
  }
  writeFileSync(`${outDir}/${name}.parts.json`, JSON.stringify({
    name, total: n, partSize: PART_SIZE, size: data.length,
  }))
  rmSync(`${outDir}/${name}`, { force: true })
}

// COOP/COEP 开启 Cross-Origin Isolation：
// 1) 允许 onnxruntime-web 使用 SharedArrayBuffer 多线程 WASM（兜底加速）
// 2) WebGPU 本身不需要，但需要安全上下文（localhost/HTTPS 天然满足）
export default defineConfig({
  // 顶层 plugins 数组：wrangler（Cloudflare）检测到 Vite 项目时会注入其插件，
  // 必须存在该数组否则报 "Cannot modify Vite config"；无副作用，不注入时为空
  plugins: [],
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
            // jsep.wasm (25.6 MiB) 超 25 MiB 限制：同源分片托管（19 MiB × 2 片 + manifest），
            // 运行时由 Worker 的 fetch 拦截下载拼接（见 src/worker.js loadJsepWasm）
            sliceFile('ort/ort-wasm-simd-threaded.jsep.wasm', 'dist/ort', 'ort-wasm-simd-threaded.jsep.wasm')
            // Vite 会把 ORT 内部的 jsep.wasm 静态引用额外打包为 assets 孤儿副本（同样超限），一并剔除
            if (existsSync('dist/assets')) {
              for (const f of readdirSync('dist/assets')) {
                if (f.startsWith('ort-wasm-simd-threaded.jsep')) rmSync(`dist/assets/${f}`, { force: true })
              }
            }
            // 模型 (165 MB) 超 25 MiB 限制：同源分片托管（19 MiB × 9 片 + manifest）
            sliceFile(MODEL_FILE, 'dist/models', 'htdemucs_fp16weights.onnx')
          },
        },
      ],
    },
  },
})
