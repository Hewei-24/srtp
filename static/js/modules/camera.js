import { API } from '../api.js';

export class CameraManager {
    constructor(videoEl, onResultCallback) {
        this.videoEl = videoEl;
        this.stream = null;
        this.interval = null;
        this.active = false;
        this.onResult = onResultCallback;
        this.currentEmotion = 'neutral';
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            this.videoEl.srcObject = this.stream;
            this.videoEl.style.display = 'block';
            this.active = true;
            return true;
        } catch (e) {
            console.error('摄像头启动失败', e);
            return false;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.videoEl.style.display = 'none';
        this.stopAnalysis();
        this.active = false;
    }

    startAnalysis() {
        if (!this.active) return;
        this.interval = setInterval(() => this.capture(), 2000);
    }

    stopAnalysis() {
        if (this.interval) clearInterval(this.interval);
    }

    async capture() {
        const canvas = document.createElement('canvas');
        canvas.width = 640; canvas.height = 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.videoEl, 0, 0, canvas.width, canvas.height);
        
        const result = await API.analyzeEmotion(canvas.toDataURL('image/jpeg', 0.7));
        if (result.success) {
            this.currentEmotion = result.dominant_emotion;
            if (this.onResult) this.onResult(result);
        }
    }

    getCurrentEmotion() {
        return this.currentEmotion;
    }
}