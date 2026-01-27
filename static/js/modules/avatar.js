import { API } from '../api.js';

export class AvatarController {
    constructor(elements) {
        this.imgEl = elements.img;
        this.idleVideoEl = elements.idle;
        this.talkVideoEl = elements.talk; // 这个 <video> 元素现在用来放无声循环视频
        this.loadingEl = elements.loading;
        
        this.audioObj = null; // 新增：用于播放 TTS 音频
        this.isIdle = false;
    }

    // 初始化：加载所选形象并启动待机
    async init() {
        const id = localStorage.getItem('selectedAvatar') || '1';
        const url = localStorage.getItem('selectedAvatarUrl') || '/avatars/avatar1.png';
        
        this.imgEl.src = url;
        
        // 告诉后端同步设置
        await API.setAvatar(id);
        this.startIdle(id);
    }

    // 启动待机模式
    async startIdle(avatarId) {
        if (this.isIdle) return;
        
        try {
            const data = await API.getIdleVideo(avatarId || localStorage.getItem('selectedAvatar'));
            if (data.success && data.videos.length > 0) {
                this.idleVideoEl.src = data.videos[0];
                this.idleVideoEl.style.display = 'block';
                this.imgEl.style.display = 'none';
                this.talkVideoEl.style.display = 'none';
                
                await this.idleVideoEl.play();
                this.isIdle = true;
            }
        } catch (e) {
            console.warn('待机视频加载失败，回退到静态图', e);
            this.showStatic();
        }
    }

    stopIdle() {
        this.isIdle = false;
        this.idleVideoEl.pause();
        this.idleVideoEl.style.display = 'none';
    }

    // 播放预置视频
    playTalking(videoUrl, audioUrl) {
    this.stopIdle();
    
    // 隐藏静态图和加载动画
    this.imgEl.style.display = 'none';
    this.loadingEl.style.display = 'none';
    
    // 1. 设置说话视频 (循环播放)
    this.talkVideoEl.src = videoUrl;
    this.talkVideoEl.style.display = 'block';
    this.talkVideoEl.loop = true;  // 关键：让视频循环
    this.talkVideoEl.muted = true; // 关键：视频本身静音
    
    // 2. 准备音频
    if (this.audioObj) {
        this.audioObj.pause();
        this.audioObj = null;
    }
    
    if (audioUrl) {
        this.audioObj = new Audio(audioUrl);
        
        // 3. 同步播放
        this.audioObj.onplay = () => {
            this.talkVideoEl.play().catch(e => console.error("视频播放失败", e));
        };
        
        // 4. 当音频结束时 -> 停止视频，回到待机
        this.audioObj.onended = () => {
            this.talkVideoEl.pause();
            this.talkVideoEl.currentTime = 0;
            setTimeout(() => this.startIdle(), 200);
        };
        
        // 启动音频
        this.audioObj.play().catch(e => {
            console.error("音频播放失败", e);
            // 如果音频播放失败，直接播放视频然后回到待机
            this.talkVideoEl.play();
            setTimeout(() => this.startIdle(), 3000);
        });
    } else {
        // 如果只有视频没有音频，就播一遍视频
        this.talkVideoEl.loop = false;
        this.talkVideoEl.play();
        this.talkVideoEl.onended = () => this.startIdle();
    }
}

    showLoading() {
        this.stopIdle();
        this.imgEl.style.display = 'block';
        this.loadingEl.style.display = 'block';
    }

    showStatic() {
        this.loadingEl.style.display = 'none';
        this.talkVideoEl.style.display = 'none';
        this.imgEl.style.display = 'block';
    }
}