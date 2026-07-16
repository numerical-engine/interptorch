import torch
import torch.nn as nn


class PCHIP(nn.Module):
    """PCHIPによる補間を行うレイヤー

    Attributes:
        x_coarse (torch.Tensor): 補間元の座標。shapeは(Nc, )。
        x_fine (torch.Tensor): 補間先の座標。shapeは(Nf, )。
    """
    def __init__(self, x_coarse:torch.Tensor, x_fine:torch.Tensor)->None:
        super().__init__()
        if not torch.all(x_coarse[1:] > x_coarse[:-1]):
            raise ValueError("x_coarse must be strictly increasing.")
        if x_fine[0] < x_coarse[0] or x_fine[-1] > x_coarse[-1]:
            raise ValueError("x_fine must be inside [x_coarse[0], x_coarse[-1]].")

        self.register_buffer("x_coarse", x_coarse)
        self.register_buffer("x_fine", x_fine)
    
    def forward(self, y:torch.Tensor)->torch.Tensor:
        """y_coarseの結果を受け取り、x_fineの座標に合わせて補間する。特徴量ごとに補間を行う。

        Args:
            y_coarse (torch.Tensor): 疎な座標における出力。shapeは(B, Nc, d)で、Bはバッチサイズ、Ncは疎な座標の数、dは特徴量の次元数。
        Returns:
            torch.Tensor: 補間結果。shapeは(B, Nf, d)。
        """
        x = self.x_coarse
        xf = self.x_fine

        h = x[1:] - x[:-1]
        delta = (y[:,1:,:] - y[:,:-1,:]) / h.view(1, -1, 1)

        nc = x.numel()
        if nc == 2:
            m = torch.stack((delta[:,0,:], delta[:,0,:]), dim = 1)
        else:
            delta_prev = delta[:,:-1,:]
            delta_next = delta[:,1:,:]
            h_prev = h[:-1].view(1, -1, 1)
            h_next = h[1:].view(1, -1, 1)

            w1 = 2.0*h_next + h_prev
            w2 = h_next + 2.0*h_prev

            same_sign = (delta_prev*delta_next) > 0
            safe_delta_prev = torch.where(same_sign, delta_prev, torch.ones_like(delta_prev))
            safe_delta_next = torch.where(same_sign, delta_next, torch.ones_like(delta_next))
            denom = (w1/safe_delta_prev) + (w2/safe_delta_next)
            m_internal = (w1 + w2)/denom
            m_middle = torch.where(same_sign, m_internal, torch.zeros_like(m_internal))

            m0 = ((2.0*h[0] + h[1])*delta[:,0,:] - h[0]*delta[:,1,:])/(h[0] + h[1])
            cond0 = (m0*delta[:,0,:]) <= 0
            cond0_alt = ((delta[:,0,:]*delta[:,1,:]) < 0) & (torch.abs(m0) > 3.0*torch.abs(delta[:,0,:]))
            m0 = torch.where(cond0, torch.zeros_like(m0), m0)
            m0 = torch.where(cond0_alt, 3.0 * delta[:,0,:], m0)

            mn = ((2.0*h[-1] + h[-2])*delta[:,-1,:] - h[-1]*delta[:,-2, :])/(h[-1] + h[-2])
            condn = (mn*delta[:,-1,:]) <= 0
            condn_alt = ((delta[:,-1,:]*delta[:,-2,:]) < 0) & (torch.abs(mn) > 3.0*torch.abs(delta[:,-1,:]))
            mn = torch.where(condn, torch.zeros_like(mn), mn)
            mn = torch.where(condn_alt, 3.0*delta[:,-1,:], mn)

            m = torch.cat((m0.unsqueeze(1), m_middle, mn.unsqueeze(1)), dim=1)

        idx = torch.searchsorted(x, xf, right=True) - 1
        idx = idx.clamp(0, nc - 2)

        xj = x[idx]
        hj = h[idx]
        t = (xf - xj)/hj

        yj = y[:,idx,:]
        yj1 = y[:,idx + 1,:]
        mj = m[:,idx,:]
        mj1 = m[:,idx + 1,:]

        t = t.view(1, -1, 1)
        hj = hj.view(1, -1, 1)
        t2 = t*t
        t3 = t2*t

        h00 = 2.0*t3 - 3.0*t2 + 1.0
        h10 = t3 - 2.0*t2 + t
        h01 = -2.0*t3 + 3.0*t2
        h11 = t3 - t2

        y_fine = h00*yj + h10*hj*mj + h01*yj1 + h11*hj*mj1
        return y_fine