import requests
import os

API_TOKEN = "17aad97300b738cfedf40b6b885a2691"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
BASE_URL = "https://civitai.com/api/v1/models"

save_dir = "civitai_sdxl_loras"
os.makedirs(save_dir, exist_ok=True)

page = 1
while True:
    params = {"types": "LORA", "limit": 100, "page": page}
    resp = requests.get(BASE_URL, headers=headers, params=params).json()
    items = resp.get("items", [])
    if not items:
        break

    for item in items:
        # look through modelVersions for SDXL base
        for version in item.get("modelVersions", []):
            # check tags or keywords if version metadata contains SDXL
            tags = version.get("metadata", {}).get("tags", [])
            if "SDXL" not in tags:
                continue

            # find a safetensors file
            for file in version.get("files", []):
                if file["name"].endswith(".safetensors"):
                    url = file["downloadUrl"]
                    fname = file["name"]
                    out_path = os.path.join(save_dir, fname)
                    if os.path.exists(out_path):
                        continue
                    print(f"Downloading {fname}")
                    with requests.get(url, headers=headers, stream=True) as dl:
                        with open(out_path, "wb") as f:
                            for chunk in dl.iter_content(1024*1024):
                                f.write(chunk)

    page += 1