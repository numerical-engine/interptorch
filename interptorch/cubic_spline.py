import torch
import torch.nn as nn

class CubicSpline(nn.Module):
    """Cubic spline による補間を行うレイヤー

    Attributes:
        x_coarse (torch.Tensor): 補間元の座標。shapeは(Nc, )。
        x_fine (torch.Tensor): 補間先の座標。shapeは(Nf, )。
    """

    def __init__(self, x_coarse: torch.Tensor, x_fine: torch.Tensor) -> None:
        super().__init__()
        if not torch.all(x_coarse[1:] > x_coarse[:-1]):
            raise ValueError("x_coarse must be strictly increasing.")
        if x_fine[0] < x_coarse[0] or x_fine[-1] > x_coarse[-1]:
            raise ValueError("x_fine must be inside [x_coarse[0], x_coarse[-1]].")

        self.register_buffer("x_coarse", x_coarse)
        self.register_buffer("x_fine", x_fine)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y_coarse の結果を受け取り、x_fine の座標に合わせて補間する。

        Args:
            y (torch.Tensor): 疎な座標における出力。shapeは(B, Nc, d) で、Bはバッチサイズ、Ncは疎な座標の数、dは特徴量の次元数。
        Returns:
            torch.Tensor: 補間結果。shapeは(B, Nf, d)。
        """
        x = self.x_coarse.to(device=y.device, dtype=y.dtype)
        xf = self.x_fine.to(device=y.device, dtype=y.dtype)

        nc = x.numel()
        if nc == 2:
            t = (xf - x[0])/(x[1] - x[0])
            t = t.view(1, -1, 1)
            return (1.0 - t)*y[:,:1,:] + t*y[:,1:,:]

        h = x[1:] - x[:-1]
        delta = (y[:,1:,:] - y[:,:-1,:])/h.view(1, -1, 1)

        system = torch.zeros((nc, nc), dtype=y.dtype, device=y.device)
        system[0, 0] = 1.0
        system[-1, -1] = 1.0

        idx = torch.arange(1, nc - 1, device=y.device)
        system[idx, idx - 1] = h[:-1]
        system[idx, idx] = 2.0 * (h[:-1] + h[1:])
        system[idx, idx + 1] = h[1:]

        rhs = torch.zeros((nc, y.shape[0], y.shape[2]), dtype=y.dtype, device=y.device)
        rhs[1:-1] = 6.0 * (delta[:,1:,:] - delta[:,:-1,:]).permute(1, 0, 2)
        rhs = rhs.reshape(nc, -1)

        m = torch.linalg.solve(system, rhs).reshape(nc, y.shape[0], y.shape[2]).permute(1, 0, 2)

        idx = torch.searchsorted(x, xf, right=True) - 1
        idx = idx.clamp(0, nc - 2)

        xj = x[idx]
        hj = h[idx]
        xj1 = x[idx + 1]

        yj = y[:,idx, :]
        yj1 = y[:,idx + 1, :]
        mj = m[:,idx, :]
        mj1 = m[:,idx + 1, :]

        xj = xj.view(1, -1, 1)
        xj1 = xj1.view(1, -1, 1)
        hj = hj.view(1, -1, 1)

        a = (xj1 - xf.view(1, -1, 1)) / hj
        b = (xf.view(1, -1, 1) - xj) / hj

        y_fine = (
            (a**3) * mj * hj**2 / 6.0
            + (b**3) * mj1 * hj**2 / 6.0
            + (yj - mj * hj**2 / 6.0) * a
            + (yj1 - mj1 * hj**2 / 6.0) * b
        )
        return y_fine