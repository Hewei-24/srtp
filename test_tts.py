# -*- coding: utf-8 -*-
import requests
import json

url = "https://api.siliconflow.cn/v1/audio/speech"
token = "sk-lvtuhfndddcmdyvnjtbzjuobfoewylsnqaqwfsnuznpilhkp"

request_data = {
                "model": "IndexTeam/IndexTTS-2",
                "voice": "IndexTeam/IndexTTS-2:claire",
                "stream": True,
                "input": "如果你现在运行第 3 步报错，把报错最后 30 行贴我；另外也把 dir checkpoints 的输出贴一下，我就能立刻定位是“模型没放对/缺依赖/ffmpeg/torch-cuda”哪个环节。",
                "max_tokens": 1600,
                "response_format": "mp3",
                "speed": 1,
                "gain": 0
                }
headers = {'Content-Type': 'application/json', 'Authorization': "Bearer " + token}
try:
    res = requests.post(url=url, data=json.dumps(request_data), headers=headers)
    if res.status_code != 200:
        print(res.text)
    with open('./test.mp3', 'wb') as file:
        file.write(res.content)
except Exception as e:
    print(request_data)
    print(e)