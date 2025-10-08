import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Init process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Important for shuffling
    for batch in dataloader:
        batch = batch.to(local_rank)
        outputs = model(batch)
        loss.backward()
        optimizer.step()
