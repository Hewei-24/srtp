# 大学生心理分析数字人代理

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.3+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/DeepFace-0.0.79+-orange.svg" alt="DeepFace">
  <img src="https://img.shields.io/badge/Three.js-r128-purple.svg" alt="Three.js">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

一个基于 AI 的大学生心理健康支持系统，结合**实时面部表情识别**与**智能心理咨询**，为大学生提供个性化的心理健康服务。

## 功能特性

### 核心功能

- **实时表情识别**: 基于 DeepFace 库，支持识别 7 种情绪（开心、悲伤、生气、恐惧、惊讶、厌恶、平静）
- **AI 心理咨询**: 集成 DeepSeek API，提供专业、温暖的心理支持和建议
- **3D 数字人交互**: 基于 Three.js 的可爱雪人形象，根据情绪实时变化表情
- **多模态分析**: 结合文字输入与面部表情，提供更精准的心理状态评估
- **对话历史管理**: 支持多轮对话，保持上下文连贯性

### 技术亮点

- 前后端分离架构，RESTful API 设计
- WebRTC 实时视频流处理
- 响应式 UI 设计，支持移动端访问
- 支持本地模型与云端 API 双模式

## 项目结构

```
srtp/
├── app.py                      # 主服务器 (Flask + DeepSeek API)
├── integrated_server.py        # 集成服务器 (支持本地模型)
├── psychological_agent.py      # 心理分析代理类
├── index.html                  # 前端主页面
├── requirements.txt            # Python 依赖
├── pyproject.toml              # UV 包管理配置
├── uv.lock                     # UV 锁定文件
├── .python-version             # Python 版本配置
├── templates/
│   └── index.html              # Flask 模板页面
└── outputs/
    └── psychology_trained_model/
        ├── psychology_training_data.json    # 训练数据
        └── psychology_data_template.json    # 数据模板
```

## 快速开始

### 环境要求

- Python 3.8 (SadTalker 依赖)
- Python 3.13+ (主项目)
- CUDA 11.8 (GPU 加速)
- 摄像头（用于表情识别）
- 现代浏览器（Chrome/Firefox/Edge）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/srtp.git
   cd srtp
   ```

2. **安装 SadTalker 依赖**

   进入 SadTalker 目录并创建 Python 3.8 虚拟环境：
   ```bash
   cd SadTalker
   uv venv --python 3.8
   .venv\Scripts\activate
   ```

   安装依赖：
   ```bash
   uv pip install dlib-bin
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   uv pip install -r requirements.txt
   ```

   **修复兼容性问题**：

   打开文件 `.venv\lib\site-packages\basicsr\data\degradations.py`，找到第 8 行，将 `functional_tensor` 改为 `functional`：
   ```python
   # 修改前
   from torchvision.transforms.functional_tensor import rgb_to_grayscale
   # 修改后
   from torchvision.transforms.functional import rgb_to_grayscale
   ```

3. **下载模型权重**

   在 `SadTalker` 目录下新建 `checkpoints` 文件夹，结构如下：
   ```
   SadTalker/
   └── checkpoints/
       ├── SadTalker_V0.0.2_.pth
       ├── mapping_00109-00224.pth
       ├── mapping_00223-00443.pth
       ├── mapping_00000-00223.pth
       └── ...
   ```

   下载地址：[SadTalker Releases](https://github.com/OpenTalker/SadTalker/releases)

4. **配置 FFmpeg**

   - 下载 [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) (release-essentials)
   - 解压后进入 `bin` 文件夹
   - 将 `ffmpeg.exe` 和 `ffprobe.exe` 复制到 `SadTalker` 项目根目录

5. **安装主项目依赖**

   回到主目录并安装依赖：
   ```bash
   cd ..
   uv pip install -r requirements.txt
   ```

6. **配置 API 密钥**

   在 `app.py` 中配置以下 API 密钥：
   ```python
   # DeepSeek API (用于 AI 对话)
   DEEPSEEK_API_KEY = "your-deepseek-api-key"

   # SiliconFlow TTS API (用于语音合成)
   TTS_API_TOKEN = "your-siliconflow-api-token"
   ```

   > 建议将 API 密钥存储在环境变量中以提高安全性

7. **启动服务**
   ```bash
   uv run app.py
   ```

8. **访问应用**

   打开浏览器访问：http://localhost:5000

## 使用指南

### 基本使用流程

1. 打开应用后，系统会询问是否开启摄像头
2. 点击「开启摄像头」按钮授权摄像头访问
3. 点击「开启表情分析」启动实时情绪检测
4. 在输入框中描述你的心理状态或问题
5. 点击「发送」获取 AI 心理咨询建议
6. 3D 数字人会根据检测到的情绪做出相应表情

### API 接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/analyze` | POST | 心理分析主接口 |
| `/api/analyze_local` | POST | 本地模型分析接口 |
| `/api/analyze_emotion` | POST | 表情识别接口 |
| `/api/model/status` | GET | 模型状态查询 |
| `/api/health` | GET | 健康检查 |
| `/api/conversation/summary` | GET | 对话摘要 |
| `/api/conversation/reset` | POST | 重置对话 |

### 请求示例

**心理分析请求：**
```json
POST /api/analyze
{
  "message": "最近学习压力很大，感觉很焦虑",
  "detected_emotion": "sad"
}
```

**响应示例：**
```json
{
  "success": true,
  "response": "我理解你的感受...",
  "detected_emotion": "sad",
  "model_source": "deepseek_api",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (index.html)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Three.js   │  │   WebRTC    │  │    用户交互界面      │  │
│  │  3D 数字人  │  │  摄像头捕获  │  │  输入/输出/状态     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     后端 (Flask Server)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  DeepFace   │  │ DeepSeek    │  │   对话历史管理       │  │
│  │  表情识别   │  │  API 调用   │  │   上下文维护        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 数据流程

```
用户输入 (文字 + 摄像头图像)
         │
         ▼
┌─────────────────┐
│  前端捕获表情    │ ──► Base64 编码图像
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ /api/analyze_   │ ──► DeepFace 分析
│    emotion      │ ──► 返回情绪分数
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  /api/analyze   │ ──► 结合文字 + 情绪
└─────────────────┘ ──► DeepSeek API
         │
         ▼
┌─────────────────┐
│   AI 响应显示   │ ──► 3D 数字人动画
└─────────────────┘
```

## 依赖说明

### Python 依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| flask | >=2.3.0 | Web 框架 |
| flask-cors | >=4.0.0 | 跨域支持 |
| requests | >=2.31.0 | HTTP 客户端 |
| deepface | >=0.0.79 | 面部表情识别 |
| opencv-python | >=4.8.0 | 图像处理 |
| numpy | >=1.24.0 | 数值计算 |
| tf-keras | >=2.20.1 | DeepFace 依赖 |

### 前端依赖

| 库名 | 版本 | 用途 |
|------|------|------|
| Three.js | r128 | 3D 图形渲染 |
| WebRTC | 原生 | 摄像头访问 |

### 可选依赖（本地模型）

| 包名 | 用途 |
|------|------|
| torch | PyTorch 深度学习框架 |
| transformers | Hugging Face 模型库 |
| peft | 参数高效微调 |

## 情绪识别说明

系统支持识别以下 7 种情绪：

| 情绪 | 英文 | 图标 | 描述 |
|------|------|------|------|
| 开心 | happy | 😊 | 积极愉快的情绪状态 |
| 悲伤 | sad | 😢 | 低落、难过的情绪 |
| 生气 | angry | 😠 | 愤怒、烦躁的情绪 |
| 恐惧 | fear | 😨 | 紧张、害怕的情绪 |
| 惊讶 | surprise | 😲 | 意外、震惊的情绪 |
| 厌恶 | disgust | 🤢 | 反感、排斥的情绪 |
| 平静 | neutral | 😐 | 中性、平稳的情绪 |

## 配置选项

### 环境变量

```bash
# DeepSeek API 配置
export DEEPSEEK_API_KEY="your-api-key"

# 服务器配置
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="5000"
export FLASK_DEBUG="true"
```

### 表情分析配置

在 `app.py` 中可调整以下参数：

```python
# 表情分析间隔（毫秒）
EMOTION_ANALYSIS_INTERVAL = 2000

# DeepFace 检测后端
DETECTOR_BACKEND = 'opencv'  # 可选: 'opencv', 'ssd', 'mtcnn', 'retinaface'
```

## 常见问题

### Q: 摄像头无法启动？
A: 请确保浏览器已授权摄像头访问权限，并检查是否有其他应用占用摄像头。

### Q: 表情识别不准确？
A: 请确保光线充足，面部正对摄像头，避免遮挡面部。

### Q: API 调用失败？
A: 请检查 DeepSeek API 密钥是否有效，网络连接是否正常。

### Q: 本地模型如何使用？
A: 运行 `integrated_server.py` 并确保已下载 Qwen1.5-0.5B 模型及训练好的适配器。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [DeepFace](https://github.com/serengil/deepface) - 面部表情识别
- [DeepSeek](https://www.deepseek.com/) - AI 对话模型
- [Three.js](https://threejs.org/) - 3D 图形库
- [Flask](https://flask.palletsprojects.com/) - Web 框架

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-username/srtp/issues)
- 发送邮件至：your-email@example.com

---

<p align="center">
  <strong>用 AI 守护每一颗年轻的心</strong>
</p>
