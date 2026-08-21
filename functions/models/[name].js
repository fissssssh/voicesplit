// Cloudflare Pages Functions：从 R2 桶流式服务模型文件
// 路由：/models/<文件名>（如 /models/htdemucs_fp16weights.onnx）
// 同源服务 → 浏览器无 CORS 问题；R2 流式响应无 100MB Worker 限制
// 需要：Pages 项目绑定 R2 桶（Binding 名 VOICESPLIT_ASSETS，见 README 部署章节）
export async function onRequestGet(context) {
  const name = context.params.name
  if (!name || name.includes('/') || name.includes('..')) {
    return new Response('Bad request', { status: 400 })
  }
  const obj = await context.env.VOICESPLIT_ASSETS.get(name)
  if (!obj) {
    return new Response('Not found', { status: 404 })
  }
  const headers = new Headers()
  obj.writeHttpMetadata(headers)
  headers.set('Content-Type', 'application/octet-stream')
  headers.set('Cache-Control', 'public, max-age=31536000, immutable')
  headers.set('ETag', obj.httpEtag)
  return new Response(obj.body, { headers })
}
