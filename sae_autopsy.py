from overcomplete.sae import TopKSAE,SAE,RATopKSAE, RAJumpSAE,QSAE,BatchTopKSAE,MpSAE

for model in [TopKSAE,SAE,RATopKSAE, RAJumpSAE,QSAE,BatchTopKSAE,MpSAE]:
    input_shape=10
    nb_concepts=10_000
    
    sae:SAE=model(input_shape=input_shape,nb_concepts=nb_concepts)
    print(model,type(sae), model.__name__)
    
    for name,module in sae.named_children():
        print(name, type(module),end="")
        if hasattr(module,"weight"):
            print(module.weight.cpu().detach().size())
        else:
            print(" ")