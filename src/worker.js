// 推理 Worker：onnxruntime-web 跑 htdemucs（WebGPU 优先，WASM 兜底）
// 与主线程只交换 Float32Array（transferable，零拷贝）
import * as ort from 'onnxruntime-web'

ort.env.wasm.wasmPaths = '/ort/'
// 多线程：ORT 默认只用到 4 线程（Math.min(4, cores/2)），这里启用全部逻辑核心。
// 必须在 wasm 实例化之前设置；依赖 COOP/COEP 隔离（SharedArrayBuffer），
// 未隔离时 ORT 自动降级单线程。
if (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) {
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency, 32)
}

const SEGMENT = 343980 // htdemucs 固定输入长度 ≈ 7.8s @ 44.1kHz
const STEM_KEYS = ['drums', 'bass', 'other', 'vocals']

// jsep.wasm (25.6 MiB) 超静态托管 25 MiB 单文件限制，从部署产物中剔除。
// 加载链：1) 本地完整文件（dev） 2) 同源分片下载拼接（部署，主路径）
//        3) jsDelivr gzip CDN 兜底（解压后返回）
// 拦截所有 jsep.wasm 请求（/ort/ 与 Vite 打包的 /assets/ 路径），返回完整 wasm Response。
// 拉取失败时 WebGPU 路径自动降级 CPU，不影响功能。
const JSEP_PARTS_URL = '/ort/ort-wasm-simd-threaded.jsep.wasm.parts.json'
const JSEP_GZ_CDN = 'https://cdn.jsdelivr.net/gh/fissssssh/voicesplit@main/ort-gz/ort-wasm-simd-threaded.jsep.wasm.gz'

/** 加载 jsep.wasm：同源分片拼接（主）→ gz CDN 解压（兜底），结果缓存 */
async function loadJsepWasm() {
  let cache = null
  try { cache = await caches.open('voicesplit-jsep-v1') } catch { /* 无缓存环境 */ }
  // 1) 缓存命中
  if (cache) {
    const hit = await cache.match(JSEP_PARTS_URL)
    if (hit) {
      const raw = await hit.arrayBuffer()
      if (raw.byteLength > 1e6) {
        return new Response(raw, { headers: { 'Content-Type': 'application/wasm' } })
      }
      await cache.delete(JSEP_PARTS_URL)
    }
  }
  // 2) 同源分片（部署时产物含 part 文件；dev 时 manifest 404）
  const manResp = await fetch(JSEP_PARTS_URL)
  if (manResp.ok) {
    const man = await manResp.json()
    const full = new Uint8Array(man.size)
    let received = 0
    for (let i = 0; i < man.total; i++) {
      const resp = await fetch(`${JSEP_PARTS_URL.replace('.parts.json', `.part${i}`)}`)
      if (!resp.ok) throw new Error(`jsep part${i} HTTP ${resp.status}`)
      const part = new Uint8Array(await resp.arrayBuffer())
      full.set(part, received)
      received += part.length
    }
    if (full[0] === 0x00 && full[1] === 0x61 && full[2] === 0x73 && full[3] === 0x6d) { // "\0asm"
      const resp = new Response(full, { headers: { 'Content-Type': 'application/wasm' } })
      if (cache) cache.put(JSEP_PARTS_URL, resp.clone())
      return resp
    }
    throw new Error('jsep 拼接后 wasm magic 校验失败')
  }
  // 3) gz CDN 兜底
  const gzResp = await fetch(JSEP_GZ_CDN)
  if (!gzResp.ok) throw new Error(`jsep gz CDN HTTP ${gzResp.status}`)
  const gz = await gzResp.arrayBuffer()
  const raw = await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
  return new Response(raw, { headers: { 'Content-Type': 'application/wasm' } })
}

const origFetch = globalThis.fetch
globalThis.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : (input && input.url) || ''
  if (url.endsWith('ort-wasm-simd-threaded.jsep.wasm') && !url.startsWith('http')) {
    // 1) 本地完整文件（dev 模式存在；部署时被 SPA HTML 顶替则继续）
    const local = await origFetch(url, init).catch(() => null)
    if (local && local.ok && !local.headers.get('Content-Type')?.includes('text/html')) {
      return local
    }
    // 2/3) 同源分片 → gz CDN 兜底
    try {
      return await loadJsepWasm()
    } catch (err) {
      return new Response(`jsep.wasm 加载失败: ${err.message}`, { status: 500 })
    }
  }
  return origFetch(input, init)
}

/** WebGPU 可用性预检：requestAdapter 返回 null（无 GPU/驱动问题）即视为不可用 */
async function detectWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false
  try {
    return !!(await navigator.gpu.requestAdapter())
  } catch {
    return false
  }
}

let session = null
let ep = null
let numThreads = null

/** 用主线程传入的模型数据（ArrayBuffer）建会话。模型加载/进度在主线程完成，只发生一次 */
async function init(modelData, opts = {}) {
  try {
    await createSession(new Uint8Array(modelData), opts)
  } catch (e) {
    postMessage({ type: 'error', message: `模型加载失败: ${e.message}` })
    throw e
  }
}

async function createSession(modelData, opts = {}) {
  // 图优化级别说明：'all' 的常量折叠会把内存峰值推过 Emscripten 32 位地址空间上限
  // （htdemucs fp16 权重 165MB，图优化时峰值可达数 GB），导致 std::bad_alloc。
  // WebGPU 路径 GPU 内存模型不同可保留 'all'；WASM 兜底用 'disabled'（已实测可稳定建会话）
  const baseOpts = { graphOptimizationLevel: 'all' }
  let webgpuErr = null
  // 预检：无 WebGPU 环境（Firefox/Safari 等）直接走 WASM，
  // 避免 jsep 模块加载失败污染全局 wasm 初始化状态
  const gpuAvailable = opts.forceWasm ? false : await detectWebGPU()

  if (gpuAvailable) {
    // 1) 首选 WebGPU（Chrome/Edge 121+，fp16 权重原生支持）
    try {
      session = await ort.InferenceSession.create(modelData, {
        ...baseOpts,
        executionProviders: ['webgpu'],
      })
      ep = 'webgpu'
      postMessage({ type: 'ready', ep, numThreads })
      return
    } catch (err) {
      webgpuErr = err
      // 标记可重试：主线程会销毁本 worker 用 WASM 重新初始化（绕过被污染的 ort 状态）
      postMessage({
        type: 'error',
        couldRetryCpu: true,
        message: `WebGPU 初始化失败，将用 CPU 重试: ${err.message}`,
      })
      throw err
    }
  }
  // 2) WASM 兜底（多线程需要 COOP/COEP，若隔离开启则自动多线程）
  try {
    session = await ort.InferenceSession.create(modelData, {
      ...baseOpts,
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
    })
    ep = 'wasm'
    numThreads = ort.env.wasm.numThreads
  } catch (wasmErr) {
    postMessage({
      type: 'error',
      message: `WASM 推理初始化失败: ${wasmErr.message}${webgpuErr ? `\nWebGPU 失败原因: ${webgpuErr.message}` : ''}`,
    })
    throw wasmErr
  }
  postMessage({ type: 'ready', ep, numThreads })
}

/** 跑一段 [1,2,343980] 的混音，返回 4 个 stem 的 (2,343980) 输出
 *  模型输出为单个 tensor `stems`，形状 [1, 4, 2, 343980]，stem 顺序固定：
 *  drums, bass, other, vocals */
async function runSegment(id, mix) {
  const t0 = performance.now()
  const tensor = new ort.Tensor('float32', mix, [1, 2, SEGMENT])
  const out = await session.run({ mix: tensor })
  const ms = performance.now() - t0
  const flat = out.stems.data // Float32Array(4 * 2 * SEGMENT)
  const stems = {}
  for (let i = 0; i < 4; i++) {
    const off = i * 2 * SEGMENT
    stems[STEM_KEYS[i]] = flat.slice(off, off + 2 * SEGMENT) // transferable
  }
  postMessage({ type: 'result', id, stems, ms }, Object.values(stems).map(s => s.buffer))
}

onmessage = async (e) => {
  try {
    const msg = e.data
    if (msg.type === 'init') {
      await init(msg.model, { forceWasm: !!msg.forceWasm })
    } else if (msg.type === 'segment') {
      await runSegment(msg.id, msg.data)
    }
  } catch (err) {
    postMessage({ type: 'error', message: err.message })
  }
}
