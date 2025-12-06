import os
import subprocess
import sys

# ================= 配置区域 =================
# 1. 在这里填入你的图片文件名 (必须放在项目根目录，或者写绝对路径)
IMAGE_PATH = "my_photo.png" 

# 2. 在这里填入你的音频文件名
AUDIO_PATH = "my_audio.mp3"

# 3. 输出文件夹名字
OUTPUT_DIR = "results"
# ===========================================

def run_sadtalker():
    # 检查文件是否存在，防止跑一半报错
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 错误：找不到图片文件 -> {IMAGE_PATH}")
        return
    if not os.path.exists(AUDIO_PATH):
        print(f"❌ 错误：找不到音频文件 -> {AUDIO_PATH}")
        return

    # === 自动检测虚拟环境 (关键修改) ===
    # 这里的逻辑是：优先寻找当前目录下的 .venv/Scripts/python.exe
    # 这样即使你忘记激活环境，脚本也会强制使用虚拟环境的 Python 来运行 SadTalker
    cwd = os.getcwd()
    venv_python = os.path.join(cwd, ".venv", "Scripts", "python.exe")
    
    if os.path.exists(venv_python):
        python_exec = venv_python
        print(f"✅ 已自动锁定虚拟环境: {venv_python}")
    else:
        # 如果找不到 .venv，就回退到使用当前环境的 python (此时需要你手动激活)
        python_exec = "python"
        print("⚠️ 未检测到 .venv 文件夹，将尝试使用默认 Python (请确保你已在终端手动激活了环境)")

    print(f"🚀 开始生成视频...")
    print(f"📷 图片: {IMAGE_PATH}")
    print(f"🎵 音频: {AUDIO_PATH}")

    # 构建命令 (针对 3060 8G 优化的参数)
    cmd = [
        python_exec, "inference.py", # 这里使用自动检测到的 python 路径
        "--driven_audio", AUDIO_PATH,
        "--source_image", IMAGE_PATH,
        "--result_dir", OUTPUT_DIR,
        "--still",              # 减少头部乱动，更稳定
        "--preprocess", "crop", # 处理整张图片，不只是脸部裁剪
        "--enhancer", "gfpgan", # 必须开启，否则人脸模糊
        "--batch_size", "5"     # 显存优化
    ]

    # 执行命令
    try:
        # 实时打印子进程输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ 成功！视频已保存到 {OUTPUT_DIR} 文件夹中。")
        else:
            print("\n❌ 生成过程中出现了错误。")
            
    except Exception as e:
        print(f"\n❌ 运行失败: {str(e)}")

if __name__ == "__main__":
    run_sadtalker()