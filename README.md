# 心晴 · 大学生心理健康辅助系统

> 基于本地大模型 + 数字人技术的大学生心理健康服务平台
>
> SRTP 项目组 · v2.4

---

## 项目简介

**心晴**是一款面向在校大学生的智能心理健康辅助系统，集成本地微调心理大模型、实时面部情绪识别、数字人视频合成与 TTS 语音合成四项核心技术，以"树洞智能体"和"心理润园"两大模块为主体，为大学生提供私密、温暖、随时可用的心理健康支持。

本项目所有推理均在本地完成，无需联网调用商业 API，保护用户隐私。

---

## 功能模块

### 🌳 树洞智能体
用户选择数字人助手后进入一对一对话界面：
- 基于 Qwen1.5-0.5B 微调的心理咨询大模型，生成专业、温暖的回复
- 实时摄像头情绪识别（DeepFace），结合面部情绪优化回复策略
- Edge TTS 语音合成，配合预置说话视频实现数字人"开口说话"
- 支持浏览器语音输入（Web Speech API）
- 多数字人形象切换，支持用户上传自定义形象

### 🌿 心理润园
心理健康科普与自助工具集：
- **科普文章**：焦虑、抑郁、压力、睡眠、人际关系、自我关怀等主题
- **心理自测**：7 题心理健康量表，按得分提供分级建议
- **呼吸放松**：4 种呼吸练习（4-7-8 / 方形 / 腹式 / 等长），带动画倒计时引导
- **资源推荐**：心理援助热线、推荐书单、冥想音频、校内咨询入口

---

## 技术架构

```
前端 (HTML / CSS / Vanilla JS ES Module)
│
├── home.html        统一首页与导航
├── select.html      数字人助手选择页
├── index.html       主对话页（树洞智能体）
└── garden.html      心理润园

后端 (Python · Flask)
│
├── app.py           主服务，RESTful API
├── 本地大模型        Qwen1.5-0.5B + LoRA 微调适配器
├── DeepFace         面部情绪识别
├── Edge TTS         中文语音合成（离线）
└── SadTalker        数字人视频驱动（可选）

static/
├── js/              前端模块（api / avatar / camera / speech）
├── speaking_videos/ 预置说话视频（avatar1_talking.mp4 等）
└── avatars/         数字人头像图片

idle_videos/         待机循环视频
models/              本地模型文件
outputs/             LoRA 适配器权重
audio_output/        TTS 生成音频（运行时生成）
uploads/             用户上传形象（运行时生成）
```

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.9 — 3.11 |
| CUDA | 建议 11.8+（CPU 也可运行，速度较慢） |
| 内存 | 建议 8 GB 以上 |
| 磁盘 | 模型文件约 2 GB |
| 浏览器 | Chrome 90+ / Edge 90+（语音功能需 HTTPS 或 localhost） |

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-org/xinqing.git
cd xinqing
```

### 2. 安装依赖

建议使用虚拟环境：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

安装 Python 依赖：

```bash
pip install flask flask-cors opencv-python numpy requests werkzeug
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate
pip install deepface
pip install edge-tts
pip install SpeechRecognition pydub
```

> **注意**：若无 GPU，将 `--index-url` 替换为 CPU 版本：
> ```bash
> pip install torch torchvision torchaudio
> ```

### 3. 准备模型文件

将 Qwen1.5-0.5B 基础模型放至：
```
models/Qwen1.5-0.5B/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model.safetensors（或分片文件）
```

将 LoRA 微调适配器放至：
```
outputs/psychology_trained_model/
├── adapter_config.json
└── adapter_model.safetensors
```

> 若无微调权重，系统会自动降级使用基础模型，功能不受影响。

### 4. 准备数字人资源

```
avatars/
├── avatar1.png     # 预置形象 1
├── avatar2.png     # 预置形象 2
└── avatar3.png     # 预置形象 3

static/speaking_videos/
├── avatar1_talking.mp4   # 形象 1 说话视频（循环播放）
├── avatar1_talking2.mp4  # 可放多个，系统随机选取
└── avatar2_talking.mp4

idle_videos/
└── idle_avatar1.mp4      # 待机视频（可选）
```

> speaking_videos 目录中每个形象可放多个说话视频，系统每次回复时随机选取一个播放。

### 5. 启动服务

```bash
python app.py
```

服务启动后访问：

| 地址 | 说明 |
|------|------|
| http://localhost:5000 | 首页（自动跳转） |
| http://localhost:5000/home | 统一导航首页 |
| http://localhost:5000/select | 选择数字人助手 |
| http://localhost:5000/index | 主对话页面 |
| http://localhost:5000/garden | 心理润园 |

---

## API 接口文档

所有接口以 `http://localhost:5000/api` 为前缀。

### 心理分析

```
POST /api/analyze
```

请求体：
```json
{
  "message": "我最近压力很大，睡不好觉",
  "detected_emotion": "sad",
  "avatar_id": "1",
  "generate_video": true
}
```

响应：
```json
{
  "success": true,
  "response": "我理解你的感受...",
  "detected_emotion": "sad",
  "model_source": "local_psychology_model",
  "audio_url": "/api/audio/tts_xxxxxxxx.mp3",
  "video_url": "/static/speaking_videos/avatar1_talking.mp4",
  "video_urls": ["/static/speaking_videos/avatar1_talking.mp4", "..."],
  "video_generated": true
}
```

### 情绪识别

```
POST /api/analyze_emotion
```

请求体：
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

响应：
```json
{
  "success": true,
  "dominant_emotion": "happy",
  "emotion_scores": {
    "happy": 85.2,
    "neutral": 10.1,
    "sad": 4.7
  }
}
```

### 数字人管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/get_avatars` | 获取全部可用形象列表 |
| POST | `/api/set_avatar` | 设置当前使用的形象 |
| POST | `/api/upload_avatar` | 上传自定义形象图片 |
| GET | `/api/idle_videos?avatar_id=1` | 获取指定形象的待机视频 |

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/model/status` | 本地模型加载状态 |
| GET | `/api/conversation/summary` | 对话摘要 |
| POST | `/api/conversation/reset` | 重置对话历史 |
| GET | `/api/debug` | 调试信息（开发用） |

---

## 项目文件结构

```
xinqing/
│
├── app.py                      # Flask 主服务（后端全部逻辑）
│
├── templates/                  # HTML 页面
│   ├── home.html               # 统一首页
│   ├── select.html             # 助手选择页
│   ├── index.html              # 主对话页
│   └── garden.html             # 心理润园
│
├── static/
│   ├── css/
│   │   └── style.css           # 全局样式（已基本内联，保留兼容）
│   ├── js/
│   │   ├── api.js              # 后端 API 封装
│   │   ├── main.js             # 主页面逻辑
│   │   ├── select.js           # 选择页逻辑
│   │   └── modules/
│   │       ├── avatar.js       # 数字人控制器
│   │       ├── camera.js       # 摄像头 & 情绪识别
│   │       └── speech.js       # 语音输入管理
│   └── speaking_videos/        # 说话视频资源
│
├── avatars/                    # 数字人头像图片
├── idle_videos/                # 待机循环视频
├── models/                     # 本地模型文件（不含于仓库）
├── outputs/                    # LoRA 适配器权重（不含于仓库）
├── audio_output/               # TTS 生成音频（运行时）
├── uploads/                    # 用户上传形象（运行时）
├── SadTalker/                  # SadTalker 子模块（可选）
│
└── README.md
```

---

## 常见问题

**Q: 启动时报 `ModuleNotFoundError`**
运行 `pip install -r requirements.txt`，或根据错误信息单独安装缺失的包。某些包（如 `deepface`、`torch`）体积较大，首次安装需要较长时间。

**Q: 情绪识别不可用 / `DeepFace` 报错**
DeepFace 首次运行会自动下载预训练权重（约 500 MB），需要联网。若网络不稳定，可提前手动下载并放置到 `~/.deepface/weights/` 目录。

**Q: 数字人嘴型不动 / 视频无法播放**
检查 `static/speaking_videos/` 目录下是否有对应 `avatar{id}_talking*.mp4` 文件，文件名中需包含 `avatar1`（或对应编号）。视频格式需为 H.264 编码的 MP4。

**Q: 语音输入按钮灰色不可用**
浏览器语音识别需要在 `localhost` 或 HTTPS 环境下运行。若在局域网其他设备访问，需为服务配置 SSL 证书，或在浏览器设置中手动开启麦克风权限。

**Q: 本地模型回复质量较差**
Qwen1.5-0.5B 是 5 亿参数的轻量模型，在没有微调权重的情况下回复质量有限。若有微调后的 LoRA 权重，将其放置到 `outputs/psychology_trained_model/` 后重启服务即可生效。

**Q: 如何增加更多说话视频**
将新视频（MP4 格式）命名为 `avatar1_talking2.mp4`、`avatar1_talking3.mp4` 等，放入 `static/speaking_videos/` 目录。系统会自动扫描目录，每次回复时随机选取一个播放。

---

## 开发说明

### 前端模块说明

前端使用原生 ES Module，无需构建工具，直接由浏览器加载。各模块职责：

- `api.js`：统一封装所有后端接口调用，修改后端地址只需改此文件的 `API_BASE_URL`
- `avatar.js`：管理数字人的静态图 / 待机视频 / 说话视频三种状态切换
- `camera.js`：摄像头启动、截帧、调用情绪识别接口
- `speech.js`：封装 Web Speech API，处理语音识别的生命周期

### 添加新数字人形象

1. 将头像图片（建议 512×512，PNG）放入 `avatars/avatar4.png`
2. 在 `app.py` 的 `get_available_avatars()` 函数中，将循环上限从 `range(1, 4)` 改为 `range(1, 5)`
3. 准备对应的说话视频 `avatar4_talking.mp4` 放入 `static/speaking_videos/`

### 修改模型参数

在 `app.py` 顶部修改：
```python
LOCAL_MODEL_PATH    = "models/Qwen1.5-0.5B"      # 基础模型路径
LOCAL_ADAPTER_PATH  = "outputs/psychology_trained_model"  # LoRA 适配器路径
```

生成参数可在 `generate_local_model_response()` 函数中调整：
```python
max_new_tokens     = 300    # 最大生成长度
temperature        = 0.7    # 采样温度（越高越发散）
top_p              = 0.9    # 核采样概率
repetition_penalty = 1.1    # 重复惩罚
```

---

## 依赖清单

```
flask
flask-cors
opencv-python
numpy
requests
werkzeug
torch
transformers
peft
accelerate
deepface
edge-tts
SpeechRecognition
pydub
```

---

## 版本历史

| 版本 | 说明 |
|------|------|
| v1.0 | 基础对话功能，调用外部 API |
| v2.0 | 集成本地 Qwen 模型，支持离线运行 |
| v2.2 | 接入 DeepFace 情绪识别，新增摄像头功能 |
| v2.3 | 支持数字人形象选择与用户自定义上传 |
| v2.4 | 接入 Edge TTS + 预置说话视频随机播放，新增心晴首页与心理润园模块 |

---

## 许可证

本项目为 SRTP 课题研究成果，仅供学术研究与学习交流使用，不得用于商业目的。

---

*如有问题，欢迎提 Issue 或联系项目组。*
