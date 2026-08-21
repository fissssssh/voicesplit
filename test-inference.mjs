// 冒烟测试（web 运行时, Node 环境）：验证模型可加载、输入输出键名/形状、推理耗时
// 注：Node 下 wasm 内存受地址空间限制，禁用图优化降低峰值内存
import fs from 'node:fs'
import * as ort from 'onnxruntime-web'

const data = fs.readFileSync('public/models/htdemucs_fp16weights.onnx')
console.log('模型文件:', (data.length / 1e6).toFixed(1), 'MB')

const t0 = Date.now()
const session = await ort.InferenceSession.create(new Uint8Array(data), {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'disabled',
})
console.log(`会话创建: ${((Date.now() - t0) / 1000).toFixed(1)}s`)

console.log('输入:', JSON.stringify(session.inputNames))
console.log('输出:', JSON.stringify(session.outputNames))

// 用 1 段随机噪声验证形状 + 测量 CPU 推理耗时
const SEGMENT = 343980
const mix = new Float32Array(2 * SEGMENT)
for (let i = 0; i < mix.length; i++) mix[i] = (Math.random() * 2 - 1) * 0.05

const t1 = Date.now()
const out = await session.run({ mix: new ort.Tensor('float32', mix, [1, 2, SEGMENT]) })
console.log(`推理耗时: ${((Date.now() - t1) / 1000).toFixed(1)}s (CPU, 1 段 ≈7.8s 音频)`)

for (const [k, v] of Object.entries(out)) {
  const flat = v.data
  let min = 1e9, max = -1e9, sum = 0
  for (let i = 0; i < flat.length; i += 977) { const x = flat[i]; if (x < min) min = x; if (x > max) max = x; sum += x }
  console.log(`输出 ${k}: shape=${JSON.stringify(v.dims)} type=${v.type} sample(min=${min.toFixed(3)} max=${max.toFixed(3)} mean=${(sum / (flat.length / 977)).toFixed(4)})`)
}
