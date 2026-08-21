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
