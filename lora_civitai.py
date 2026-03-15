import requests
import os

SAVE_DIR="loras"
os.makedirs(SAVE_DIR,exist_ok=True)

page=1

while True:

    url=f"https://civitai.com/api/v1/models?types=LORA&page={page}"
    data=requests.get(url).json()

    if len(data["items"])==0:
        break

    for model in data["items"]:
        for version in model["modelVersions"]:
            for file in version["files"]:

                if file["name"].endswith(".safetensors"):
                    download_url=file["downloadUrl"]
                    fname=file["name"]

                    path=os.path.join(SAVE_DIR,fname)

                    if not os.path.exists(path):
                        print("downloading",fname)
                        r=requests.get(download_url)

                        with open(path,"wb") as f:
                            f.write(r.content)

    page+=1