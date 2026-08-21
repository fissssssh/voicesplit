import { defineConfig } from 'vite'
import { cpSync, rmSync, readdirSync, existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'

// 模型分片：165MB 超静态托管 25 MiB 单文件限制，构建时切成 19 MiB/片
// （留足余量），运行时同源下载拼接（见 src/main.js loadModelParts）
const MODEL_FILE = 'public/models/htdemucs_fp16weights.onnx'
const PART_SIZE = 19 * 1024 * 1024

function sliceModel(outDir) {
  if (!existsSync(MODEL_FILE)) return
  const data = readFileSync(MODEL_FILE)
  const n = Math.ceil(data.length / PART_SIZE)
  const dir = `${outDir}/models`
  mkdirSync(dir, { recursive: true })
  for (let i = 0; i < n; i++) {
    const part = data.subarray(i * PART_SIZE, Math.min((i + 1) * PART_SIZE, data.length))
    writeFileSync(`${dir}/htdemucs_fp16weights.onnx.part${i}`, part)
  }
  writeFileSync(`${dir}/htdemucs_fp16weights.onnx.parts.json`, JSON.stringify({
    name: 'htdemucs_fp16weights.onnx', total: n, partSize: PART_SIZE, size: data.length,
  }))
  rmSync(`${dir}/htdemucs_fp16weights.onnx`, { force: true })
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
            // jsep.wasm (25.6 MiB) 超 Cloudflare 静态托管单文件 25 MiB 限制：
            // 从产物中剔除，运行时由 Worker 内的 fetch 拦截改走 CDN（GitHub Release）
            rmSync('dist/ort/ort-wasm-simd-threaded.jsep.wasm', { force: true })
            // Vite 会把 ORT 内部的 jsep.wasm 静态引用额外打包为 assets 孤儿副本（同样超限），一并剔除
            if (existsSync('dist/assets')) {
              for (const f of readdirSync('dist/assets')) {
                if (f.startsWith('ort-wasm-simd-threaded.jsep')) rmSync(`dist/assets/${f}`, { force: true })
              }
            }
            // 模型 (165 MB) 超 25 MiB 限制：改为同源分片托管（19 MiB × 9 片 + manifest），
            // 运行时下载拼接；不再删除 models 目录
            sliceModel('dist')
          },
        },
      ],
    },
  },
})
