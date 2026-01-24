# check_my_files.py
import os
import json

# 你的路径
adapter_path = r"D:\Study\srtp\3ndweek\srtp\outputs\psychology_trained_model"

print("🔍 检查你的适配器文件")
print("=" * 60)

if os.path.exists(adapter_path):
    print(f"✅ 路径存在: {adapter_path}")
    print("\n📄 文件列表:")

    files = os.listdir(adapter_path)
    for file in files:
        file_path = os.path.join(adapter_path, file)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  - {file} ({size_kb:.1f} KB)")

        # 如果是 JSON 文件，读取内容
        if file.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    print(f"    类型: {file}")
                    if file == 'adapter_config.json':
                        base_model = content.get('base_model_name_or_path', '未知')
                        print(f"    基础模型: {base_model}")
                    print(f"    内容预览: {str(content)[:200]}...")
            except Exception as e:
                print(f"    读取失败: {e}")
else:
    print(f"❌ 路径不存在: {adapter_path}")

print("=" * 60)