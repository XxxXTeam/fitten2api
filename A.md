架构概述
该扩展是完全混淆的单文件 webpack bundle (extension/dist/extension.js，~9.5MB)，使用自定义字符串数组 + 十六进制偏移解析器混淆。没有 source map 或 TypeScript 源码。
AI 接口接入方式
1. 后端地址
- 主 AI 服务: https://fc.fittenlab.cn (配置项: fittencode.server.serverURL)
- 云存储: https://api.fittentech.com (配置项: fittencode.cloudStorage.serverUrl)
2. 通信方式
- HTTP 客户端: undici (Node.js 高性能 HTTP 库)
- 通过 llm.a... 命令注册 LLM API 通信通道
- API 路径包含 /api/project_..._flags/ 等模式
3. 请求格式
使用自定义模板引擎，Prompt 格式为：
<|system|>
系统提示
<|end|>
<|user|>
用户消息
<|end|>
<|assistant|>
模型回复
<|end|>
4. 功能接口
- Chat: 对话补全（streaming，maxTokens=1024）
- Inline Completion: 自动补全（基于上下文实时生成）
- Task Engine: explain-code, generate-unit-test, diagnose-errors, optimize-code, find-bugs, document-code 等，各自有独立 prompt 模板
5. 模板文件 위치
template/chat/ 和 template/task/ 下的 .rdt.md 文件定义了各功能的 prompt 结构和参数（maxTokens、stop tokens、variables等）。
6. 认证
通过 accessToken + refreshToken + userId 机制认证，token 存储在 VS Code 的 secrets 中。