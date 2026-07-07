# 生成描述文本

import requests
import pandas as pd
import time

# ====== 2. 读取药物数据 ======
# 假设你的药物文件叫 drug_names_pandas.csv，包含一列 drug_name
df = pd.read_csv("E:\others\MF\CD\ceshi-1\drug_names.csv")

# 确保有 drug_name 列
if "drug_name" not in df.columns:
    raise ValueError("CSV 文件中缺少 'drug_name' 列")

# ====== 3. 为每个药物生成描述 ======
descriptions = []
failed_drugs = []

for index, drug in enumerate(df["drug_name"]):
    try:
        # 更具体的提示词，要求100字英文描述
        payload = {
            "model": "deepseek-ai/DeepSeek-V3.1",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert pharmacologist. Generate concise and accurate English drug descriptions of approximately 100 words."
                },
                {
                    "role": "user",
                    "content": f"Generate a 100-word English description of the drug '{drug}'. Include: 1) drug class/mechanism, 2) main indications, 3) key characteristic. Be concise and professional."
                }
            ],
            "max_tokens": 200,  # 限制输出长度
            "temperature": 0.3   # 较低的温度值，使输出更确定性
        }

        print(f"Processing [{index+1}/{len(df)}]: {drug}")
        
        # 发送 POST 请求到硅基流动的 API
        response = requests.post(url, json=payload, headers=headers, timeout=60)

        # 获取模型的响应
        if response.status_code == 200:
            res = response.json()
            description = res["choices"][0]["message"]["content"].strip()
            # 进一步确保长度控制
            words = description.split()
            if len(words) > 120:  # 如果超过40个词，截断
                description = ' '.join(words[:100]) + "..."
        else:
            description = None
            print(f"API Error for {drug}: {response.status_code} - {response.text}")
        
    except Exception as e:
        description = None
        print(f"Error processing {drug}: {e}")

    if description:
        descriptions.append(description)
        print(f"✓ {drug} -> {description[:50]}...")  # 只显示前50个字符
    else:
        descriptions.append("")  # 生成失败则添加空字符串
        failed_drugs.append(drug)
        print(f"✗ Failed to generate description for {drug}")
    
    # 添加延迟，避免请求过于频繁
    time.sleep(1)

# ====== 4. 保存结果 ======
df["description"] = descriptions
output_file = "drug_with_descriptions.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"生成完成，结果已保存到 {output_file}")

# 显示统计信息
success_count = len([d for d in descriptions if d and not d.startswith("Error")])
print(f"\n统计信息:")
print(f"成功生成: {success_count}/{len(df)}")
print(f"失败: {len(failed_drugs)}")

if failed_drugs:
    print(f"\n失败的药物:")
    for drug in failed_drugs:
        print(f"  - {drug}")
    
    # 保存失败列表
    failed_df = pd.DataFrame({"failed_drug": failed_drugs})
    failed_df.to_csv("failed_drugs.csv", index=False)
    print("失败药物列表已保存到 failed_drugs.csv")
