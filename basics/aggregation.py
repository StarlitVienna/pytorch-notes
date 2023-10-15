import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")


x = torch.arange(0, 100+10, 10) #default dtype for arange is long

print(x)
print(f"X min --> {torch.min(x)}")
print(f"X max --> {torch.max(x)}")
print(f"X sum --> {torch.sum(x)}")
print(f"X mean --> {torch.mean(x.type(torch.float32))}") # there are 11 items
print(f"X min along a certain axis --> {x.argmin(dim=0)}") # Will return the index position, not the value
print(f"X max along a certain axis --> {x.argmax(dim=0)}") # Will return the index position, not the value