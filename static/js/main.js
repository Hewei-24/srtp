import { API } from './api.js';
import { AvatarController } from './modules/avatar.js';
import { CameraManager } from './modules/camera.js';
import { SpeechManager } from './modules/speech.js';

document.addEventListener('DOMContentLoaded', () => {
    // 1. 初始化数字人控制
    const avatar = new AvatarController({
        img: document.getElementById('avatar-img'),
        idle: document.getElementById('idle-video'),
        talk: document.getElementById('avatar-video'),
        loading: document.getElementById('video-loading'),
        container: document.getElementById('scene-container')
    });
    avatar.init();

    // 2. 初始化UI元素
    const ui = {
        input: document.getElementById('user-input'),
        sendBtn: document.getElementById('submit-btn'),
        responseBox: document.getElementById('response-content'),
        statusText: document.getElementById('status-text'),
        emotionResult: document.getElementById('emotion-result'),
        emotionBars: document.getElementById('emotion-bars')
    };

    // 3. 摄像头模块
    const camera = new CameraManager(
        document.getElementById('camera-preview'),
        (result) => updateEmotionUI(result)
    );

    document.getElementById('camera-toggle').onclick = async function() {
        if (camera.active) {
            camera.stop();
            this.textContent = '📷 开启摄像头';
            document.querySelector('.emotion-display').style.display = 'none';
        } else {
            const success = await camera.start();
            if (success) {
                this.textContent = '🚫 关闭摄像头';
                document.querySelector('.emotion-display').style.display = 'block';
                camera.startAnalysis();
            }
        }
    };

    // 4. 语音模块
    const speech = new SpeechManager(
        document.getElementById('voice-btn'),
        document.getElementById('voice-status'),
        (text) => {
            ui.input.value = text;
            setTimeout(handleAnalyze, 1000); // 1秒后自动发送
        }
    );
    document.getElementById('voice-btn').onclick = () => speech.toggle();

    // 5. 交互逻辑
    ui.sendBtn.onclick = handleAnalyze;
    ui.input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleAnalyze();
    });

    // 辅助函数：处理发送分析
    async function handleAnalyze() {
    const text = ui.input.value.trim();
    if (!text) return;

    // 获取当前选中的 avatar_id
    const currentId = localStorage.getItem('selectedAvatar') || '1';
    console.log('发送分析请求:', { text, avatar_id: currentId });
    
    ui.sendBtn.disabled = true;
    ui.input.disabled = true;
    ui.statusText.textContent = '正在分析心理状态...';
    
    avatar.showLoading();

    try {
        console.log('调用 API.analyzePsychology...');
        const result = await API.analyzePsychology({
            message: text,
            detected_emotion: camera.getCurrentEmotion(),
            avatar_id: currentId,
            generate_video: true
        });

        console.log('API 响应:', result);
        
        if (result.success) {
            // 显示文字
            const sourceText = result.model_source === 'local_psychology_model' ? '🧠 本地大模型' : '🤖 备用模型';
            ui.responseBox.innerHTML = `
                <p>${result.response}</p>
                <div class="response-meta">
                    ${sourceText} | 情绪: ${result.detected_emotion}
                </div>
            `;

            // 播放视频（优先使用候选列表以支持随机播放）
            if (result.video_generated && (result.video_urls || result.video_url)) {
                const videoSource = result.video_urls || result.video_url;
                console.log('播放视频（候选列表）:', videoSource, '音频:', result.audio_url);
                avatar.playTalking(videoSource, result.audio_url);
            } else {
                console.warn('视频生成失败:', result.error);
                avatar.showStatic();
            }
        } else {
            console.error('API 返回失败:', result.error);
            ui.responseBox.textContent = '分析失败: ' + (result.error || '未知错误');
            avatar.showStatic();
        }
    } catch (e) {
        console.error("分析请求失败:", e);
        ui.responseBox.innerHTML = `
            <p style="color:var(--accent-color)">系统繁忙，请稍后再试。</p>
            <div class="response-meta">错误: ${e.message}</div>
        `;
        avatar.showStatic();
    } finally {
        ui.sendBtn.disabled = false;
        ui.input.disabled = false;
        ui.input.value = '';
        ui.input.focus();
        ui.statusText.textContent = '系统就绪';
    }
}

    // 辅助函数：更新情绪UI
    function updateEmotionUI(data) {
        const map = {
            'angry': '😠 生气', 'disgust': '🤢 厌恶', 'fear': '😨 恐惧',
            'happy': '😊 开心', 'sad': '😢 悲伤', 'surprise': '😲 惊讶', 'neutral': '😐 平静'
        };
        ui.emotionResult.textContent = map[data.dominant_emotion] || data.dominant_emotion;
        
        // 渲染进度条 (简化版)
        const sorted = Object.entries(data.emotion_scores).sort((a,b) => b[1]-a[1]).slice(0, 3);
        ui.emotionBars.innerHTML = sorted.map(([k, v]) => `
            <div style="margin:5px 0; font-size:12px">
                <div style="display:flex;justify-content:space-between"><span>${map[k]||k}</span><span>${v.toFixed(0)}%</span></div>
                <div style="background:#eee;height:4px;border-radius:2px"><div style="width:${v}%;background:var(--secondary-color);height:100%"></div></div>
            </div>
        `).join('');
    }
});