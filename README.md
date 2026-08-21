# VoiceSplit — 浏览器内人声分离

纯前端、零后端的歌曲人声 / 伴奏分离。上传音频后由 **htdemucs**（Meta 开源最强分离模型之一）在本机完成分离——**音频数据全程不出设备**，无上传、无 Cookie、无追踪。

![深色模式](https://img.shields.io/badge/theme-dark%20%26%20light-0F0F23) ![Tech](https://img.shields.io/badge/WebGPU-%2322C55E) ![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ 特性

- **🎤 双轨输出**：人声 + 伴奏（伴奏 = 鼓 + 贝斯 + 其他，htdemucs 4-stem 引擎合并）
- **⚡ GPU 优先**：WebGPU 加速（Chrome/Edge 121+），无 GPU 自动降级 CPU 全核多线程
- **🔒 完全本地**：推理、解码、编码全部在浏览器内完成，歌曲不离开你的设备
- **🎨 双主题**：OLED 深黑 / 浅色主题，跟随系统偏好，手动切换记忆在本地
- **🔍 环境诊断面板**：启动即显示平台、逻辑核心数、隔离状态、GPU 名称与 fp16 支持
- **📊 平滑进度**：假定目标 + 指数渐近逼近的连续进度条（段内无真实进度的黑盒补偿）
- **📦 零依赖部署**：模型随仓库（Git LFS），静态托管即开即用

## 🧰 技术栈

| 组件 | 方案 |
|---|---|
| 推理 | [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.27（**WebGPU** 优先，WASM 自动兜底，全核心多线程） |
| 模型 | [StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx) fp16 权重（4-stem: 人声/鼓/贝斯/其他） |
| 音频解码/重采样 | Web Audio API（`decodeAudioData` + `OfflineAudioContext`，无需 ffmpeg） |
| 处理策略 | 7.8s 分段推理 + 25% 重叠线性窗 Overlap-Add（与官方推理一致） |
| 前端 | Vite 7 + 原生 JS（Web Worker 隔离推理线程，Transferable 零拷贝传输） |
| 设计 | 手写设计系统（Righteous + Poppins，CSS 变量双主题，SVG 图标，`prefers-reduced-motion`） |

## 🚀 快速开始

```bash
npm install        # 安装依赖（国内网络可加 --registry=https://registry.npmmirror.com）
npm run dev        # http://localhost:5173
```

**浏览器要求**：

| 浏览器 | WebGPU | WASM 多线程 | WASM 单线程 |
|---|---|---|---|
| Chrome / Edge 121+ | ✅ 首选 | ✅ | ✅ |
| Firefox / Safari | ❌ | ✅（需隔离头） | ✅ |

- WebGPU 要求安全上下文（localhost / HTTPS）与 2016 年后的 GPU（含 `shader-f16`，如 Intel Arc、RTX 20+、Apple Silicon）
- WASM 多线程需要 COOP/COEP 响应头（本仓库 dev/preview/部署配置均已包含）

## 🖱 使用

1. 打开页面，诊断面板显示本机能力（核心数 / 隔离 / GPU 型号 / fp16）
2. 拖入音频（MP3 / WAV / FLAC / OGG / M4A，建议 ≤10 分钟）
3. 等待分离（状态区显示：解码 → 加载模型 → 初始化引擎 → 平滑进度百分比）
4. 试听 / 下载人声、伴奏两轨 WAV

## 🏗 架构

```
上传 → decodeAudioData 解码 → OfflineAudioContext 重采样 44.1kHz
     → 343980 样本分段（stride 257985，零填充补齐）
     → 模型加载（主线程一次，缓存复用）
     → Web Worker: onnxruntime-web 推理（WebGPU → WASM 多线程 → 单线程兜底）
     → 官方线性窗重叠相加 → 4 stem Float32（drums/bass/other/vocals）
     → 合并 2 轨：人声 + 伴奏（= drums+bass+other 求和，立即释放原始数据）
     → 试听（AudioContext 即时播放 + 进度条）/ 下载（按需编码 16-bit WAV）
```

关键设计：

- **EP 降级链**：主线程预检 `requestAdapter()` → WebGPU → WASM（多线程，`hardwareConcurrency` 全核）→ 单线程；降级无"失败"噪音
- **模型缓存**：165MB 模型只在首次加载（带进度），之后歌曲复用；WebGPU 降级重试不重复加载
- **多线程前提**：COOP/COEP 隔离（`vite.config.js` + `public/_headers` 已配置）
- **WASM 图优化**：CPU 路径用 `disabled` 级别，规避 Emscripten 32 位内存上限的 `bad_alloc`（GPU 路径保留 `all`）

## ⏱ 性能预期（4 分钟歌曲）

| 后端 | 耗时 |
|---|---|
| WebGPU（Arc 130T 级核显） | 约 20–60 秒 |
| WASM 多线程（8-16 核） | 约 1–3 分钟 |
| WASM 单线程 | 3–5 分钟 |

内存峰值约 0.5–1GB（取决于歌曲时长）。

## 📦 模型来源与替换

模型随仓库托管（Git LFS，`public/models/htdemucs_fp16weights.onnx`，165MB）。
原始出处（国内需镜像）：

```
https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx
```

替换模型：放入 `public/models/` 并修改 `src/main.js` 的 `MODEL_URL`。
模型契约（`node test-inference.mjs` 实测验证）：

- 输入 `mix: [1, 2, 343980]`（float32，44.1kHz 双声道）
- 输出单个 tensor `stems: [1, 4, 2, 343980]`，顺序 drums, bass, other, vocals

## ☁️ 部署

静态托管即可（产物已含模型与 ORT 运行时，约 200MB）：

| 平台 | 多线程加速 | 说明 |
|---|---|---|
| **Cloudflare Pages** | ✅ | `_headers` 已配置，构建 `npm run build`，输出 `dist` |
| **Netlify** | ✅ | 同样读取 `_headers` |
| **Vercel** | ✅ | 需在 vercel.json 配置 COOP/COEP 头 |
| GitHub Pages | ❌ 单线程 | 无法配置响应头，功能不受影响，只是慢 |

**必须 HTTPS**（WebGPU 要求安全上下文）。

### 模型 CDN（免费，零配置）

165MB 模型超静态托管单文件 25 MiB 限制，默认经 **GitHub LFS 媒体服务** 分发
（`media.githubusercontent.com`，免费、CORS 已实测开放，模型本就随仓库 LFS 托管）：
- 运行时模型源顺序：本地 `/models/` → LFS 媒体 CDN → hf-mirror → GitHub Release（兜底）
- 首次访问从 CDN 下载 165MB（有进度条），之后浏览器 Cache API 缓存，无需重复下载
- LFS 免费额度 1GB 存储 + 1GB/月带宽，个人/小规模使用足够

可选：如需完全同源分发（更好速度与稳定性），可用 **Cloudflare R2 桶 + Pages Functions**（免费额度 10GB 存储）：
1. Cloudflare Dashboard → **R2** → 创建桶 → 上传 `htdemucs_fp16weights.onnx`
2. Pages 项目 → **Settings → Bindings** → R2 bucket 绑定，变量名 `VOICESPLIT_ASSETS`
3. 重新部署后 `/models/...` 由 `functions/models/[name].js` 同源返回（dev 模式自动回退本地文件）

## 📁 目录结构

```
├── index.html             # 页面结构（分步指示器 / 主题切换 / 结果卡）
├── vite.config.js         # COOP/COEP 头 + 构建时复制 ORT 运行时
├── public/
│   ├── models/            # htdemucs fp16 模型（Git LFS）
│   └── _headers           # Cloudflare Pages / Netlify 响应头
├── ort/                   # onnxruntime-web 运行时（.mjs/.wasm，构建时复制）
└── src/
    ├── main.js            # 编排：上传→解码→分段→推理→重叠相加→结果
    ├── worker.js          # 推理 Worker（WebGPU/WASM 双后端 + 降级链）
    ├── audio.js           # 音频管线（解码/重采样/分段/OLA/WAV 编码）
    └── style.css          # 设计系统（双主题令牌 / 动画 / 响应式）
```

## 📜 License

MIT

---

*Made with ❤️ — 音频永远留在你的设备上。*
