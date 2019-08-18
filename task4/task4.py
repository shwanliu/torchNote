import torch
import torch.nn as nn

optimizer = torch.optim.Adadelta(net.parameters(), lr=1e-2)