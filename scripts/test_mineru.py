import os

import dotenv
import requests

dotenv.load_dotenv()

# ==================== MinerU 配置 ====================

token = os.getenv("MINERU_API_KEY")
url = "https://mineru.net/api/v4/file-urls/batch"

# 本地文件路径
file_path = "/home/chase/study/camel-agent/download_files/基于大模型智能体的智能对话系统的研究与实现_揭会顺.pdf"

# 自动获取文件名
file_name = os.path.basename(file_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}",
}

data = {
    "files": [
        {
            "name": file_name,
            "data_id": "abcd",
        }
    ],
    "model_version": "vlm",
}


try:
    # ==================== 1. 申请上传地址 ====================

    response = requests.post(
        url,
        headers=headers,
        json=data,
    )

    if response.status_code != 200:
        print(
            f"申请上传地址失败: "
            f"status={response.status_code}, "
            f"response={response.text}"
        )
        raise SystemExit

    result = response.json()
    print(result)

    # MinerU 业务状态判断
    if result.get("code") != 0:
        print(f"申请上传地址失败: {result.get('msg')}")
        raise SystemExit

    # 批量提取任务 ID
    batch_id = result["data"]["batch_id"]

    # 文件上传地址列表
    file_urls = result["data"]["file_urls"]

    if not file_urls:
        print("MinerU 没有返回文件上传地址")
        raise SystemExit

    # 当前只上传一个文件，所以直接取第一个 URL
    upload_url = file_urls[0]

    print(f"申请上传地址成功")
    print(f"batch_id: {batch_id}")


    # ==================== 2. 上传文件 ====================

    with open(file_path, "rb") as f:
        upload_response = requests.put(
            upload_url,
            data=f,
        )

    if upload_response.status_code == 200:
        print(f"文件上传成功: {file_name}")
    else:
        print(
            f"文件上传失败: "
            f"status={upload_response.status_code}, "
            f"response={upload_response.text}"
        )


except FileNotFoundError:
    print(f"文件不存在: {file_path}")

except requests.RequestException as err:
    print(f"网络请求失败: {err}")

except Exception as err:
    print(f"发生异常: {err}")
