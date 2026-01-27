const API_BASE_URL = 'http://localhost:5000/api';

export const API = {
    // 获取数字人列表
    async getAvatars() {
        const res = await fetch(`${API_BASE_URL}/get_avatars`);
        return await res.json();
    },

    // 设置后端数字人
    async setAvatar(avatarId) {
        return fetch(`${API_BASE_URL}/set_avatar`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ avatar_id: avatarId })
        }).then(res => res.json());
    },

    // 获取待机视频
    async getIdleVideo(avatarId) {
        return fetch(`${API_BASE_URL}/idle_videos?avatar_id=${avatarId}`).then(res => res.json());
    },

    // 表情分析
    async analyzeEmotion(imageData) {
        return fetch(`${API_BASE_URL}/analyze_emotion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        }).then(res => res.json());
    },

    // 核心心理分析
    // 核心心理分析
async analyzePsychology(payload) {
    console.log('调用 analyzePsychology:', payload);
    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const result = await response.json();
    console.log('analyzePsychology 响应:', result);
    return result;
},

    // 图片上传
    async uploadAvatar(file) {
        const formData = new FormData();
        formData.append('avatar', file);
        return fetch(`${API_BASE_URL}/upload_avatar`, {
            method: 'POST',
            body: formData
        }).then(res => res.json());
    }
};