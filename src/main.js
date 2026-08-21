// 主线程：文件上传 → 解码/重采样 → 分段 → Worker 推理（WebGPU/WASM）→ 重叠相加 → 试听/下载
import {
  SAMPLE_RATE, SEGMENT, STRIDE, STEM_KEYS,
  decodeFile, resampleTo44100, segmentBuffer, createOverlapAdder, mixBacking, encodeWav,
} from './audio.js'

// 模型源（按优先级）：本地（dev/自托管）→ hf-mirror（有 CORS，实测可用）→ GitHub Release 兜底
// 165MB 模型不随部署产物（超静态托管 25 MiB 单文件限制），运行时下载并缓存
const MODEL_SOURCES = [
  '/models/htdemucs_fp16weights.onnx',
  'https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx',
  'https://github.com/fissssssh/voicesplit/releases/download/v0.1.0/htdemucs_fp16weights.onnx',
]
const MODEL_CACHE = 'voicesplit-models-v3'

// ---------- DOM ----------
const $ = (id) => document.getElementById(id)
const dropzone = $('dropzone'), fileInput = $('file-input')
const statusEl = $('status'), statusText = $('status-text'), statusDetail = $('status-detail'), barFill = $('bar-fill'), barEl = statusEl.querySelector('.bar')
const resultsEl = $('results'), cardsEl = $('cards'), fileNameEl = $('file-name')
const epBadge = $('ep-badge')

/** 平滑进度：假定目标略高于真实进度 + 指数渐近逼近（无限趋近、永不越界）
 *  推理本身是黑盒，段内无真实进度：每段完成后把目标设为 真实进度+超前量，
 *  显示值以 v += (target-v)*0.06 持续逼近，进度条全程连续移动；
 *  收尾时（目标=100）转线性补满，避免卡在 99.9% */
let smoothPct = 0, targetPct = 0, smoothRAF = null
let segProgressBase = '' // 非空时 = 分离阶段：文字百分比跟随平滑进度每帧更新
function setTargetPct(p) {
  targetPct = p
  if (!smoothRAF) smoothRAF = requestAnimationFrame(animateProgress)
}
function animateProgress() {
  const diff = targetPct - smoothPct
  if (targetPct >= 100 && diff < 2) {
    // 收尾：目标已到 100，线性补满（指数逼近永远到不了，需强制收尾）
    smoothPct += 0.25
    if (smoothPct >= 100) {
      smoothPct = 100
      barFill.style.width = '100%'
      if (segProgressBase) statusText.textContent = `${segProgressBase} 100%`
      smoothRAF = null
      return
    }
  } else if (diff > 0.001) {
    // 指数渐近：无限逼近假定目标（永不越界、持续移动）
    smoothPct += diff * 0.06
  } else {
    smoothPct = targetPct // 防御：目标回退时直接对齐
  }
  barFill.style.width = Math.min(smoothPct, 100) + '%'
  // 分离阶段：文字百分比跟随平滑值（与进度条同步，不跳变）
  if (segProgressBase) statusText.textContent = `${segProgressBase} ${smoothPct.toFixed(1)}%`
  smoothRAF = requestAnimationFrame(animateProgress)
}

/** WebGPU 可用性预检（主线程）：无适配器直接走 CPU，避免 worker 尝试失败再重试 */
async function detectWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false
  try {
    return !!(await navigator.gpu.requestAdapter())
  } catch {
    return false
  }
}

// ---------- 状态 ----------
let worker = null
let modelBuffer = null // 模型 ArrayBuffer 缓存：只加载一次，之后复用（含 WebGPU 失败重试）
let trackData = null // { vocals, backing, drums, bass, other } -> Float32Array(2 * total)
let totalSamples = 0
let songBaseName = 'song'
let processing = false

/** ONNX ModelProto 校验：protobuf 首字段必为 field 1 varint（0x08 = ir_version），
 *  且体积远大于错误页——防 SPA fallback 返回的 index.html（HTTP 200）被当模型使用 */
function looksLikeOnnx(buf) {
  return buf.length > 1e6 && buf[0] === 0x08
}

/** 加载模型到内存：多源尝试（本地 → hf-mirror → GitHub Release），
 *  下载与缓存命中均校验字节有效性；无效数据（如 SPA 回退的 HTML）不缓存并换源 */
async function loadModel(onProgress) {
  if (modelBuffer) return modelBuffer
  let cache = null
  try { cache = await caches.open(MODEL_CACHE) } catch { /* 非安全上下文时无缓存 */ }
  const diag = [] // 各源诊断，全部失败时输出
  for (const url of MODEL_SOURCES) {
    try {
      if (cache) {
        const hit = await cache.match(url)
        if (hit) {
          const cached = new Uint8Array(await hit.arrayBuffer())
          if (looksLikeOnnx(cached)) {
            modelBuffer = cached.buffer
            return modelBuffer
          }
          await cache.delete(url) // 缓存坏数据：清除
          diag.push(`${url}: 缓存数据无效(${cached.length}B, 首字节 0x${cached[0]?.toString(16)})`)
        }
      }
      const resp = await fetch(url)
      if (!resp.ok) { diag.push(`${url}: HTTP ${resp.status}`); continue }
      // 快速路径：SPA fallback / 拦截页会返回 HTML，无需下载即可跳过
      if (resp.headers.get('Content-Type')?.includes('text/html')) {
        diag.push(`${url}: 返回 HTML（SPA 回退或拦截页）`)
        continue
      }
      const total = Number(resp.headers.get('Content-Length') || 0)
      const reader = resp.body.getReader()
      const chunks = []
      let received = 0
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        received += value.length
        if (total) onProgress?.(received, total)
      }
      const buf = new Uint8Array(received)
      let off = 0
      for (const c of chunks) {
        buf.set(c, off)
        off += c.length
      }
      if (!looksLikeOnnx(buf)) { // 拿到非模型字节：不缓存，换源
        diag.push(`${url}: 数据无效(${buf.length}B, 首字节 0x${buf[0]?.toString(16)})`)
        continue
      }
      modelBuffer = buf.buffer
      if (cache) cache.put(url, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream' } }))
      return modelBuffer
    } catch (err) {
      diag.push(`${url}: ${err.message}`)
    }
  }
  throw new Error(`模型加载失败：所有源均不可用\n${diag.join('\n')}`)
}

// ---------- 上传 ----------
dropzone.addEventListener('click', () => fileInput.click())
dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click() } })
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover') })
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'))
dropzone.addEventListener('drop', (e) => {
  e.preventDefault(); dropzone.classList.remove('dragover')
  const f = e.dataTransfer.files[0]
  if (f) startSplit(f)
})
fileInput.addEventListener('change', () => { if (fileInput.files[0]) startSplit(fileInput.files[0]) })

function setStatus(text, detail = '', pct = 0) {
  statusEl.classList.remove('hidden')
  statusText.textContent = text
  statusDetail.textContent = detail
  setTargetPct(pct)
}

/** 分步指示器：n=步骤序号，state=active|done；之前的步骤自动 done */
function setStep(n, state) {
  document.querySelectorAll('.step-dot').forEach((el, i) => {
    const idx = i + 1
    el.classList.toggle('active', idx === n && state === 'active')
    el.classList.toggle('done', (idx === n && state === 'done') || idx < n)
  })
  document.querySelectorAll('.stepper .line').forEach((line, i) => {
    line.classList.toggle('on', i + 1 < n || (i + 1 === n && state === 'done'))
  })
}

const fmtDur = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

function showError(msg) {
  segProgressBase = '' // 停止平滑文字接管
  if (smoothRAF) { cancelAnimationFrame(smoothRAF); smoothRAF = null }
  let el = document.querySelector('.error')
  if (!el) {
    el = document.createElement('div')
    el.className = 'error'
    statusEl.after(el)
  }
  el.textContent = msg
  setStatus('处理失败', '', 0)
}

// ---------- 主流程 ----------
async function startSplit(file) {
  if (processing) return
  processing = true
  resultsEl.classList.add('hidden')
  document.querySelector('.error')?.remove()
  songBaseName = file.name.replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]/g, '_')
  dropzone.style.opacity = '.45'
  // 重置平滑进度（新歌曲从头开始）
  smoothPct = 0; targetPct = 0; segProgressBase = ''
  if (smoothRAF) { cancelAnimationFrame(smoothRAF); smoothRAF = null }
  barFill.style.width = '0%'

  try {
    // 1. 解码 + 重采样（浏览器原生，无需 ffmpeg）
    setStep(2, 'active')
    setStatus('解码音频…', file.name)
    const raw = await decodeFile(file)
    const audio = await resampleTo44100(raw)
    setStatus('切分音频段…', `${Math.round(audio.duration)}s @ 44.1kHz`)

    // 2. 分段
    const { segments, totalSamples: total, nCh } = segmentBuffer(audio)
    totalSamples = total
    const adder = createOverlapAdder(total, nCh)
    const nSeg = segments.length

    // 3. 加载模型（仅首次真正读取；WebGPU 重试时直接复用缓存，进度只出现一次）
    setStatus('加载模型…', '', 0)
    const model = await loadModel((received, total) => {
      const pct = (received / total * 100).toFixed(1)
      setStatus('加载模型…', `${(received / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB (${pct}%)`, received / total * 100)
    })

    // 4. 初始化推理引擎（建会话时长不定 → 缓冲条动画，避免停在 100% 的"卡顿"感）
    setStatus('初始化推理引擎…', '首次约 5–20 秒，请稍候')
    barEl.classList.add('indeterminate')
    if (!worker) worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
    const initWorker = (forceWasm) => new Promise((resolve, reject) => {
      const onMsg = (e) => {
        const m = e.data
        if (m.type === 'ready') { worker.removeEventListener('message', onMsg); resolve(m) }
        else if (m.type === 'error') {
          worker.removeEventListener('message', onMsg)
          const err = new Error(m.message)
          err.couldRetryCpu = !!m.couldRetryCpu
          reject(err)
        }
      }
      worker.addEventListener('message', onMsg)
      worker.postMessage({ type: 'init', model, forceWasm })
    })
    let initResult
    try {
      // 主线程预检 GPU：无适配器直接走 CPU，全程不出现"失败"提示
      initResult = await initWorker(!(await detectWebGPU()))
    } catch (err) {
      if (err.couldRetryCpu) {
        // GPU 路径不可用：销毁当前 worker（ORT 状态已被污染），温和降级 CPU 重建
        worker.terminate()
        worker = null
        setStatus('正在切换到 CPU 模式…', 'GPU 路径不可用，CPU 同样能完成分离')
        worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
        initResult = await initWorker(true)
      } else {
        throw err
      }
    } finally {
      barEl.classList.remove('indeterminate')
    }
    const { ep, numThreads } = initResult
    renderEpBadge(ep, numThreads)

    // 4. 逐段推理 + 重叠相加（串行保序；Worker 内消息队列天然实现流水线）
    // 分离进度从 0 干净起步（清除模型加载阶段的进度残值），文字由平滑动画接管
    smoothPct = 0; targetPct = 0
    if (smoothRAF) { cancelAnimationFrame(smoothRAF); smoothRAF = null }
    barFill.style.width = '0%'
    segProgressBase = '分离中…'
    statusText.textContent = '分离中… 0%'
    statusDetail.textContent = `${ep.toUpperCase()}${ep === 'wasm' && numThreads ? ` · ${numThreads} 线程` : ' 加速'}`
    setTargetPct(0)
    for (let i = 0; i < nSeg; i++) {
      const seg = segments[i]
      const result = await new Promise((resolve, reject) => {
        const onMsg = (e) => {
          const m = e.data
          if (m.type === 'result' && m.id === i) { worker.removeEventListener('message', onMsg); resolve(m) }
          else if (m.type === 'error') { worker.removeEventListener('message', onMsg); reject(new Error(m.message)) }
        }
        worker.addEventListener('message', onMsg)
        worker.postMessage({ type: 'segment', id: i, data: seg.data }, [seg.data.buffer])
      })
      for (const key of STEM_KEYS) adder.add(key, seg.start, result.stems[key])
      const realPct = (i + 1) / nSeg * 100
      // 假定目标 = 真实进度 + 超前量（不足一个段，上限 5%）：进度条持续逼近但不过度虚报
      const lead = Math.min(100 / nSeg * 0.8, 5)
      setTargetPct(Math.min(realPct + lead, 100))
      // 文字由平滑动画接管，这里只更新详情
      statusDetail.textContent = `${ep.toUpperCase()}${ep === 'wasm' && numThreads ? ` · ${numThreads} 线程` : ''} · 平均 ${result.ms.toFixed(0)}ms/段`
    }

    // 5. 组装结果（伴奏 = 鼓+贝斯+其他，求和后即可释放三轨原始数据）
    const stems = adder.finalize()
    trackData = {
      vocals: stems.vocals,
      backing: mixBacking(stems, total),
    }
    renderResults()
    setStep(3, 'done')
    segProgressBase = '' // 停止平滑文字接管，交回 setStatus
    setStatus('完成', `${nSeg} 段 · ${fmtDur(totalSamples / SAMPLE_RATE)} · ${ep.toUpperCase()}`, 100)
  } catch (err) {
    showError(err.message || String(err))
  } finally {
    processing = false
    dropzone.style.opacity = '1'
  }
}

function renderEpBadge(ep, numThreads) {
  const gpu = ep === 'webgpu'
  epBadge.textContent = gpu ? '⚡ GPU 加速 (WebGPU)' : `🧠 CPU 推理 (WASM${numThreads ? ` ×${numThreads}线程` : ''})`
  epBadge.className = `badge ${gpu ? 'gpu' : 'cpu'}`
}

// ---------- 环境诊断 ----------
async function renderEnvInfo() {
  const el = $('env-check')
  const parts = []
  parts.push(`<span>🖥 ${navigator.platform || '未知平台'}</span>`)
  parts.push(`<span>🧮 ${navigator.hardwareConcurrency || '?'} 逻辑核心</span>`)
  parts.push(`<span>🔒 ${self.crossOriginIsolated ? '<span class="ok">隔离已开启</span>（多线程可用）' : '<span class="bad">隔离未开启</span>（单线程）'}</span>`)
  try {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter()
      if (adapter) {
        const ai = adapter.info || {}
        const name = `${ai.description || ai.vendor || 'GPU'}${ai.architecture ? ` (${ai.architecture})` : ''}`
        // 检查半精度计算特性：htdemucs 是 fp16 模型，ORT WebGPU 依赖 shader-f16
        let fp16 = '未知'
        try {
          const dev = await adapter.requestDevice()
          fp16 = dev.features.has('shader-f16') ? '支持 ✅' : '不支持 ❌'
        } catch {
          fp16 = '设备创建失败 ❌'
        }
        parts.push(`<span>⚡ <span class="ok">WebGPU 可用</span>：${name} · fp16 计算：${fp16}${fp16.startsWith('不支持') ? '（模型为 fp16，将使用 CPU）' : ''}</span>`)
      } else {
        parts.push(`<span>⚡ <span class="bad">WebGPU 无适配器</span>（将使用 CPU）</span>`)
      }
    } else {
      parts.push(`<span>⚡ <span class="bad">浏览器不支持 WebGPU</span>（将使用 CPU）</span>`)
    }
  } catch {
    parts.push(`<span>⚡ <span class="bad">WebGPU 检测失败</span>（将使用 CPU）</span>`)
  }
  el.innerHTML = parts.join(' ')
  el.classList.remove('hidden')
}

// ---------- 结果展示 ----------
// SVG 图标（Lucide 风格，内联）
const SVG_WRAP = (inner) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`
const ICON_VOCALS = '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>'
const ICON_BACKING = '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>'
const ICON_PLAY = '<path d="m6 4 14 8-14 8Z"/>'
const ICON_PAUSE = '<path d="M6 4h4v16H6zM14 4h4v16h-4z"/>'
const ICON_DL = '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>'

const TRACKS = [
  { key: 'vocals', label: '人声', icon: ICON_VOCALS },
  { key: 'backing', label: '伴奏', icon: ICON_BACKING },
]

function renderResults() {
  fileNameEl.textContent = `— ${songBaseName}`
  cardsEl.innerHTML = ''
  for (const t of TRACKS) {
    const card = document.createElement('div')
    card.className = 'card'
    card.dataset.key = t.key
    const secs = totalSamples / SAMPLE_RATE
    const sizeMB = secs * 176.4 // 16-bit 双声道 WAV ≈ 176.4 KB/s
    card.innerHTML = `
      <div class="track-head">
        <span class="track-icon">${SVG_WRAP(t.icon)}</span>
        <span class="track-name">${t.label}</span>
        <span class="track-meta">${fmtDur(secs)} · ${sizeMB.toFixed(0)} MB</span>
      </div>
      <div class="track-progress" aria-hidden="true"><div class="fill"></div></div>
      <div class="track-ops">
        <button class="btn play" data-key="${t.key}" aria-label="试听${t.label}">${SVG_WRAP(ICON_PLAY)}<span>试听</span></button>
        <button class="btn dl" data-key="${t.key}" aria-label="下载${t.label} WAV">${SVG_WRAP(ICON_DL)}<span>下载</span></button>
      </div>`
    card.querySelector('.play').addEventListener('click', () => togglePlay(t.key))
    card.querySelector('.dl').addEventListener('click', () => downloadTrack(t.key, t.label))
    cardsEl.appendChild(card)
  }
  resultsEl.classList.remove('hidden')
}

// ---------- 试听（AudioContext 即时播放 + 进度条） ----------
let audioCtx = null, sourceNode = null, playingKey = null
let progressRAF = null, playStart = 0

function togglePlay(key) {
  if (playingKey === key) {
    stopPlay()
    return
  }
  if (sourceNode) sourceNode.stop()
  audioCtx ||= new AudioContext()
  const data = trackData[key]
  const buf = audioCtx.createBuffer(2, totalSamples, SAMPLE_RATE)
  for (let ch = 0; ch < 2; ch++) buf.copyToChannel(data.subarray(ch * totalSamples, (ch + 1) * totalSamples), ch)
  const src = audioCtx.createBufferSource()
  src.buffer = buf
  src.connect(audioCtx.destination)
  src.onended = () => stopPlay()
  src.start()
  sourceNode = src
  playingKey = key
  playStart = audioCtx.currentTime
  syncPlayButtons()
  updateProgress()
}

function updateProgress() {
  const fill = document.querySelector(`.card[data-key="${playingKey}"] .track-progress .fill`)
  if (fill) {
    const p = Math.min(1, (audioCtx.currentTime - playStart) / (totalSamples / SAMPLE_RATE))
    fill.style.width = `${p * 100}%`
  }
  progressRAF = requestAnimationFrame(updateProgress)
}

function stopPlay() {
  cancelAnimationFrame(progressRAF)
  sourceNode?.stop()
  sourceNode = null
  playingKey = null
  document.querySelectorAll('.track-progress .fill').forEach((f) => { f.style.width = '0%' })
  syncPlayButtons()
}

function syncPlayButtons() {
  document.querySelectorAll('.play').forEach((b) => {
    const on = b.dataset.key === playingKey
    b.classList.toggle('playing', on)
    b.innerHTML = `${SVG_WRAP(on ? ICON_PAUSE : ICON_PLAY)}<span>${on ? '停止' : '试听'}</span>`
  })
}

// ---------- 下载 ----------
function downloadTrack(key, label) {
  const t0 = performance.now()
  const blob = encodeWav(trackData[key], SAMPLE_RATE, 2)
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${songBaseName}_${label.replace(/[^\w一-龥]/g, '')}.wav`
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000)
  console.log(`编码 ${label}: ${(performance.now() - t0).toFixed(0)}ms, ${(blob.size / 1e6).toFixed(1)} MB`)
}

// 页面加载即渲染环境诊断
renderEnvInfo()

// ---------- 主题切换（初始值由 index.html 头部内联脚本设置，避免闪烁） ----------
$('theme-toggle')?.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'
  document.documentElement.dataset.theme = next
  try { localStorage.setItem('voicesplit-theme', next) } catch { /* 隐私模式等场景忽略 */ }
})

// 页面关闭时清理
window.addEventListener('beforeunload', () => worker?.terminate())
