// 音频处理管线：解码 → 重采样 → 分段 → 重叠相加（Overlap-Add）→ WAV 编码
// 全程使用 Float32Array，避免不必要的拷贝

export const SAMPLE_RATE = 44100 // htdemucs 要求 44.1kHz
export const SEGMENT = 343980 // 模型固定输入长度 ≈ 7.8s @ 44.1kHz
export const OVERLAP = 0.25 // demucs 官方重叠比
export const STRIDE = Math.round(SEGMENT * (1 - OVERLAP)) // 257985，相邻段起点间距

export const STEM_KEYS = ['drums', 'bass', 'other', 'vocals']

// 与官方 htdemucs 推理一致的重叠窗：中间恒为 1，首尾 overlap 个样本线性淡入/淡出
// （配合 overlap=25% 的 stride，用窗平方和归一化精确重建）
export function olaWindow(n) {
  const overlap = Math.floor(n / 4)
  const w = new Float32Array(n).fill(1)
  for (let i = 0; i < overlap; i++) {
    const f = i / overlap
    w[i] = f
    w[n - 1 - i] = f
  }
  return w
}

/** 用浏览器原生解码器解任意格式（mp3/aac/ogg/wav/flac…） */
export async function decodeFile(file) {
  const buf = await file.arrayBuffer()
  const ctx = new AudioContext()
  try {
    return await ctx.decodeAudioData(buf)
  } finally {
    await ctx.close()
  }
}

/** 重采样到 44.1kHz（OfflineAudioContext 走浏览器原生高质量重采样器） */
export async function resampleTo44100(buffer) {
  if (buffer.sampleRate === SAMPLE_RATE) return buffer
  const nCh = Math.min(buffer.numberOfChannels, 2)
  const len = Math.ceil((buffer.duration * SAMPLE_RATE))
  const ctx = new OfflineAudioContext(nCh, len, SAMPLE_RATE)
  const src = ctx.createBufferSource()
  src.buffer = buffer
  src.connect(ctx.destination)
  src.start(0)
  return ctx.startRendering()
}

/** 把整首歌切成 [1, 2, 343980] 的段（不足补零），返回 { segments, totalSamples, nCh: 2 }
 *  模型固定 2 声道输入：单声道源复制到左右声道，全部段统一双声道 */
export function segmentBuffer(buffer) {
  const srcCh = Math.min(buffer.numberOfChannels, 2)
  const nCh = 2
  const total = srcCh === 1 ? SAMPLE_RATE * buffer.duration : 2 * SAMPLE_RATE * buffer.duration
  const segments = []
  for (let start = 0; start < total; start += STRIDE) {
    const data = new Float32Array(nCh * SEGMENT) // 零填充即补零
    for (let ch = 0; ch < srcCh; ch++) {
      const src = buffer.getChannelData(ch)
      const off = ch * SEGMENT
      for (let i = 0; i < SEGMENT; i++) {
        const srcIdx = start + i
        if (srcIdx < src.length) data[off + i] = src[srcIdx]
      }
    }
    if (srcCh === 1) data.copyWithin(SEGMENT, 0, SEGMENT) // 单声道 → 复制到右声道
    segments.push({ start, data })
  }
  return { segments, totalSamples: Math.ceil(total), nCh }
}

/** 重叠相加累加器：把每个段的推理结果按窗加权叠加回全长 */
export function createOverlapAdder(totalSamples, nCh) {
  const win = olaWindow(SEGMENT)
  // 预计算窗平方和 denom[k] = Σ_seg win[i]²（重叠区归一化因子）
  const denom = new Float32Array(totalSamples)
  for (let start = 0; start < totalSamples; start += STRIDE) {
    for (let i = 0; i < SEGMENT; i++) {
      const k = start + i
      if (k < totalSamples) denom[k] += win[i] * win[i]
    }
  }
  const accs = {} // stemKey -> Float32Array(nCh * totalSamples)
  for (const k of STEM_KEYS) accs[k] = new Float32Array(nCh * totalSamples)

  return {
    accs,
    /** segOut: Float32Array(nCh * SEGMENT)，某一段的某个 stem 输出 */
    add(stemKey, start, segOut) {
      const acc = accs[stemKey]
      for (let ch = 0; ch < nCh; ch++) {
        const outOff = ch * totalSamples
        const inOff = ch * SEGMENT
        for (let i = 0; i < SEGMENT; i++) {
          const k = start + i
          if (k < totalSamples) acc[outOff + k] += segOut[inOff + i] * win[i]
        }
      }
    },
    /** 归一化并返回每 stem 的 interleaved Float32Array（顺便释放累加器） */
    finalize() {
      const out = {}
      for (const k of STEM_KEYS) {
        const a = accs[k]
        for (let ch = 0; ch < nCh; ch++) {
          const base = ch * totalSamples
          for (let i = 0; i < totalSamples; i++) {
            const d = denom[i]
            if (d > 1e-8) a[base + i] /= d
          }
        }
        out[k] = a
      }
      return out
    },
  }
}

/** 伴奏 = 鼓 + 贝斯 + 其他（htdemucs 4-stem 的标准混法），返回新数组 */
export function mixBacking(stems, totalSamples) {
  const n = 2 * totalSamples
  const out = new Float32Array(n)
  for (const k of ['drums', 'bass', 'other']) {
    const s = stems[k]
    for (let i = 0; i < n; i++) out[i] += s[i]
  }
  return out
}

/** 16-bit PCM WAV 编码（纯 JS，无依赖） */
export function encodeWav(interleaved, sampleRate, channels) {
  const n = interleaved.length
  const buf = new ArrayBuffer(44 + n * 2)
  const dv = new DataView(buf)
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  dv.setUint32(4, 36 + n * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  dv.setUint32(16, 16, true)
  dv.setUint16(20, 1, true) // PCM
  dv.setUint16(22, channels, true)
  dv.setUint32(24, sampleRate, true)
  dv.setUint32(28, sampleRate * channels * 2, true)
  dv.setUint16(32, channels * 2, true)
  dv.setUint16(34, 16, true)
  writeStr(36, 'data')
  dv.setUint32(40, n * 2, true)
  let o = 44
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, interleaved[i]))
    dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    o += 2
  }
  return new Blob([buf], { type: 'audio/wav' })
}
