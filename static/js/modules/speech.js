export class SpeechManager {
    constructor(btnEl, statusEl, onResultCallback) {
        this.btn = btnEl;
        this.status = statusEl;
        this.callback = onResultCallback;
        this.recognition = null;
        this.isListening = false;
        
        this.init();
    }

    init() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.btn.disabled = true;
            this.status.textContent = '浏览器不支持语音';
            return;
        }
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.lang = 'zh-CN';
        this.recognition.continuous = false;

        this.recognition.onstart = () => {
            this.isListening = true;
            this.btn.classList.add('listening');
            this.status.textContent = '正在聆听...';
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this.btn.classList.remove('listening');
            if (this.status.textContent === '正在聆听...') {
                this.status.textContent = '点击开始语音';
            }
        };

        this.recognition.onresult = (e) => {
            const transcript = e.results[0][0].transcript;
            this.status.textContent = `识别: ${transcript}`;
            if (this.callback) this.callback(transcript);
        };
    }

    toggle() {
        if (this.isListening) this.recognition.stop();
        else this.recognition.start();
    }
}