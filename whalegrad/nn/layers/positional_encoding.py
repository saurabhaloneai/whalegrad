import numpy as np

class RoPE:
    """Implements the rotary positional encoding [1].

    The traditional implementation rotates consecutive pairs of elements in the
    feature dimension while the default implementation rotates pairs with
    stride half the feature dimensions for efficiency.

    [1]: https://arxiv.org/abs/2104.09864

    Args:
        dims (int): The feature dimensions to be rotated. If the input feature
            is larger than dims then the rest is left unchanged.
        traditional (bool, optional): If set to True choose the traditional
            implementation which is slightly less efficient. Default: ``False``
        base (float, optional): The base used to compute angular frequency for
            each dimension in the positional encodings. Default: ``10000``
    """

    def __init__(self, dims: int, traditional: bool = False, base: float = 10000):
        self.dims = dims
        self.traditional = traditional
        self.base = base

    def _extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}"

    def _compute_rope(self, costheta, sintheta, x):
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = np.concatenate([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = np.concatenate([rx1, rx2], axis=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            raise NotImplementedError(
                "RoPE doesn't implement partial traditional application"
            )

        rx = np.concatenate([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = np.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=self.base, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return np.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int, D: int, offset: int = 0, base: float = 10000, dtype=np.float32
    ):
        D = D // 2
        positions = np.arange(offset, N, dtype=dtype)
        freqs = np.exp(-np.arange(0.0, D, dtype=dtype) * (np.log(base) / D))
        theta = np.reshape(positions, (-1, 1)) * np.reshape(freqs, (1, -1))
        return np.cos(theta), np.sin(theta)
