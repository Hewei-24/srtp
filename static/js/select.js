import { API } from './api.js';

document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('preset-avatars');
    const uploadInput = document.getElementById('file-input');
    
    // 1. 加载所有形象
    loadAvatars();

    async function loadAvatars() {
        const data = await API.getAvatars();
        if(data.success) {
            renderList(data.avatars);
        }
    }

    function renderList(list) {
        container.innerHTML = list.map(av => `
            <div class="glass-panel avatar-card" onclick="selectAvatar('${av.id}', '${av.image_url}', '${av.name}')">
                <img src="${av.image_url}" alt="${av.name}">
                <h3 style="margin-top:10px;font-size:1rem">${av.name}</h3>
            </div>
        `).join('');
    }

    // 2. 选择逻辑 (暴露给全局以便 HTML onclick 调用，或者用事件代理)
    window.selectAvatar = (id, url, name) => {
        // UI 反馈
        document.querySelectorAll('.avatar-card').forEach(el => el.classList.remove('selected'));
        event.currentTarget.classList.add('selected');

        // 保存到本地
        localStorage.setItem('selectedAvatar', id);
        localStorage.setItem('selectedAvatarName', name);
        localStorage.setItem('selectedAvatarUrl', url);

        // 启用继续按钮
        const btn = document.getElementById('continue-btn');
        btn.disabled = false;
        btn.textContent = `作为 ${name} 开始咨询`;
    };

    // 3. 上传逻辑
    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if(!file) return;
        
        const btn = document.querySelector('.upload-btn span');
        btn.textContent = '上传中...';
        
        try {
            const res = await API.uploadAvatar(file);
            if(res.success) {
                alert('上传成功！');
                loadAvatars(); // 刷新列表
            } else {
                alert('上传失败: ' + res.error);
            }
        } catch(err) {
            console.error(err);
        } finally {
            btn.textContent = '上传自定义形象';
        }
    });
});