from dino_extract import get_last_hidden_states, dino_model,dino_processor
from PIL import Image
import os
import faiss
import numpy as np

embeddings=[]
metadata=[]
limit=10000

folder="synthetic-sana"

for z,path in enumerate([f for f in os.listdir(folder) if f.endswith("png")]):
    if z>limit:
        break
    emb=get_last_hidden_states(Image.open(os.path.join(folder,path)),dino_processor,dino_model)[:,0,:].flatten().cpu().detach().numpy()
    
    metadata.append({"path":path})
    
    embeddings.append(emb)
    
embeddings=np.stack(embeddings)

dim = embeddings.shape[1]

print(embeddings.shape)

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Search each vector for exact duplicates
D, I = index.search(embeddings, k=embeddings.shape[0])  # search all vectors

# Find indices where distance is zero (or very close)
invalid=set()
duplicates = {}
for i, distances in enumerate(D):
    if i in invalid:
        continue
    for j, dist in zip(I[i], distances):
        if i != j and dist < 1e-6:  # distance almost zero
            duplicates.setdefault(i, []).append(j)
            invalid.add(j)

print("Duplicate pairs (indices):", duplicates)