import numpy as np
import torch
import torch.nn as nn

class ODECell(nn.Module):

    def __init__(self, num_units):
        super(ODECell, self).__init__()
        self.num_units = num_units

    @property
    def state_size(self):
        return self.num_units, self.num_units

    def zero_state(self, batch_size, dtype):
        x_0 = torch.zeros([batch_size, self.num_units], dtype=dtype)
        v_0 = torch.zeros([batch_size, self.num_units], dtype=dtype)
        return x_0, v_0


class BouncingODECell(ODECell):

    def __init__(self, num_units):
        super(BouncingODECell, self).__init__(num_units)
        self.dt = nn.Parameter(torch.tensor(0.3), requires_grad=False)

    def forward(self, poss, vels):
        poss = torch.split(poss, 2, 1)
        vels = torch.split(vels, 2, 1)
        poss = list(poss)
        vels = list(vels)
        for i in range(5):
            poss[0] = poss[0] + self.dt/5 * vels[0]
            poss[1] = poss[1] + self.dt/5 * vels[1]

            for j in range(2):
                vels[j] = torch.where(poss[j] + 2 > 32, -vels[j], vels[j])
                vels[j] = torch.where(0.0 > poss[j] - 2, -vels[j], vels[j])
                poss[j] = torch.where(poss[j] + 2 > 32, 32 - (poss[j] + 2 - 32) - 2, poss[j])
                poss[j] = torch.where(0.0 > poss[j] - 2, -(poss[j] - 2) + 2, poss[j])

        poss = torch.cat(poss, dim=1)
        vels = torch.cat(vels, dim=1)
        return poss, vels


class SpringODECell(ODECell):

    def __init__(self, num_units):
        super(SpringODECell, self).__init__(num_units)
        self.dt = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(np.log(1.0)), requires_grad=True)
        self.equil = nn.Parameter(torch.tensor(np.log(1.0)), requires_grad=True)

    def forward(self, poss, vels):
        poss = torch.split(poss, 2, 1)
        vels = torch.split(vels, 2, 1)
        for i in range(5):
            norm = torch.sqrt(torch.abs(torch.sum((poss[0] - poss[1]) ** 2, dim=-1, keepdim=True)))
            direction = (poss[0] - poss[1]) / (norm + 1e-4)
            F = torch.exp(self.k) * (norm - 2 * torch.exp(self.equil)) * direction
            vels[0] = vels[0] - self.dt / 5 * F
            vels[1] = vels[1] + self.dt / 5 * F

            poss[0] = poss[0] + self.dt / 5 * vels[0]
            poss[1] = poss[1] + self.dt / 5 * vels[1]

        poss = torch.cat(poss, dim=1)
        vels = torch.cat(vels, dim=1)
        return poss, vels


class GravityODECell(ODECell):

    def __init__(self, num_units):
        super(GravityODECell, self).__init__(num_units)
        self.dt = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.g = nn.Parameter(torch.tensor(np.log(1.0)), requires_grad=True)
        self.m = nn.Parameter(torch.tensor(np.log(1.0)), requires_grad=False)
        self.A = torch.exp(self.g) * torch.exp(2 * self.m)

    def forward(self, poss, vels):
        for i in range(5):
            vecs = [poss[:, 0:2] - poss[:, 2:4], poss[:, 2:4] - poss[:, 4:6], poss[:, 4:6] - poss[:, 0:2]]
            norms = [torch.sqrt(torch.clamp(torch.sum(vec ** 2, dim=-1, keepdim=True), 1e-1, 1e5)) for vec in vecs]
            F = [(vec / torch.pow(torch.clamp(norm, 1, 170), 3)) for vec, norm in zip(vecs, norms)]
            F = [(F[0] - F[2]), (F[1] - F[0]), (F[2] - F[1])]
            F = [-(self.A * f) for f in F]
            F = torch.cat(F, dim=1)
            vels = vels + self.dt / 5 * F
            poss = poss + self.dt / 5 * vels

        poss = torch.cat(poss, dim=1)
        vels = torch.cat(vels, dim=1)
        return poss, vels
