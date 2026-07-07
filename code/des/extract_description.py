
# Generate single-drug pharmacological description texts.
#
# This helper queries a language model API for drug descriptions that are later
# encoded as the single-drug semantic modality.
import requests
import os
import pandas as pd
import time

# ====== 1. Configure the SiliconFlow API ======
url = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.environ.get("SILICONFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set the SILICONFLOW_API_KEY environment variable before running this script.")
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# ====== 2. Load drug names ======
# The input CSV should contain a column named drug_name.
df = pd.read_csv("drug_names.csv")

if "drug_name" not in df.columns:
    raise ValueError("The CSV file must contain a 'drug_name' column.")


# ====== 3. Generate one description for each drug ======
descriptions = []
failed_drugs = []

for index, drug in enumerate(df["drug_name"]):
    try:
        # Use a detailed prompt to request a concise English description.
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
            "max_tokens": 200,  # Limit output length.
            "temperature": 0.3   # Use a low temperature for deterministic output.
        }

        print(f"Processing [{index+1}/{len(df)}]: {drug}")
        
        # Send the POST request to the API.
        response = requests.post(url, json=payload, headers=headers, timeout=60)

        # Parse the model response.
        if response.status_code == 200:
            res = response.json()
            description = res["choices"][0]["message"]["content"].strip()
            # Further control the description length.
            words = description.split()
            if len(words) > 120:  # Truncate overly long outputs.
                description = ' '.join(words[:100]) + "..."
        else:
            description = None
            print(f"API Error for {drug}: {response.status_code} - {response.text}")
        
    except Exception as e:
        description = None
        print(f"Error processing {drug}: {e}")

    if description:
        descriptions.append(description)
        print(f"OK {drug} -> {description[:50]}...")  # Show only the first 50 characters.
    else:
        descriptions.append("")  # Store an empty string if generation fails.
        failed_drugs.append(drug)
        print(f"Failed to generate description for {drug}")
    
    # Add a short delay to avoid overly frequent requests.
    time.sleep(1)


# ====== 4. Save results ======
df["description"] = descriptions
output_file = "drug_with_descriptions.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Generation completed. Results saved to {output_file}")


# Display summary statistics.
success_count = len([d for d in descriptions if d and not d.startswith("Error")])
print(f"\nSummary:")
print(f"Successful generations: {success_count}/{len(df)}")
print(f"Failed generations: {len(failed_drugs)}")

if failed_drugs:
    print(f"\nFailed drugs:")
    for drug in failed_drugs:
        print(f"  - {drug}")
    
    # Save the failed drug list for manual checking.
    failed_df = pd.DataFrame({"failed_drug": failed_drugs})
    failed_df.to_csv("failed_drugs.csv", index=False)
    print("Failed drug list saved to failed_drugs.csv")
