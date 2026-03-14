from dino_extract import get_last_hidden_states
from PIL import Image
import os
import faiss
import numpy as np

embeddings=[]
limit=10

folder="synthetic-sana"

for z,path in [f for f in os.listdir(folder) if f.endswith("png")]:
    if z>limit:
        break
    emb=get_last_hidden_states(Image.open(os.path.join(folder,path))).cpu().detach().numpy()
    
    embeddings.append(emb)
    
embeddings=np.stack(embeddings)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)