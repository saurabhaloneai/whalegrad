import numpy as np
from typing import Optional

class RoPE:
    

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


#-----------------------------#


class SinusoidalPositionalEncoding:
    

    def __init__(
        self,
        dims: int,
        min_freq: float = 0.0001,
        max_freq: float = 1,
        scale: Optional[float] = None,
        cos_first: bool = False,
        full_turns: bool = False,
    ):
        one_zero = 1 - np.arange(0, dims // 2) / (dims // 2 - 1)
        min_freq = np.log(min_freq)
        max_freq = np.log(max_freq)

        # Start with underscore so it is not included in the parameters
        self._sigmas = np.exp(one_zero * (max_freq - min_freq) + min_freq)
        if full_turns:
            self._sigmas = self._sigmas * (2 * np.pi)

        # Save some constants that define the implementation
        self.scale = scale or (2 / dims) ** 0.5
        self.cos_first = cos_first

    def __call__(self, x):
        y = x[..., None] * self._sigmas
        cosy = np.cos(y)
        siny = np.sin(y)

        if self.cos_first:
            y = np.hstack([cosy, siny])
        else:
            y = np.hstack([siny, cosy])

        if self.scale != 1:
            y = y * self.scale

        return y
   


class ALiBi:
    @staticmethod
    def create_alibi_matrix(
        q_sequence_length: int,
        k_sequence_length: int,
        num_heads: int,
        offset: int,
        dtype=np.float32,
    ):
        x1 = np.arange(offset, q_sequence_length)
        x2 = np.arange(0, k_sequence_length)
        distance_matrix = -np.abs(
            np.expand_dims(x1[:, None] - x2[None, :], axis=(0, 1))
        )
        alibi_slope = ALiBi.create_alibi_slope(num_heads=num_heads)
        alibi_mask = (distance_matrix * alibi_slope).astype(dtype)
        return alibi_mask

    @staticmethod
    def create_alibi_slope(num_heads):
        x = (2**8) ** (1 / num_heads)
        out = np.power(x, -np.arange(1, num_heads + 1))
        return np.expand_dims(out, axis=(-1, -2))

    def __call__(self, attention_scores, offset=0, mask=None):
        alibi_mask = ALiBi.create_alibi_matrix(
            q_sequence_length=attention_scores.shape[-2] + offset,
            k_sequence_length=attention_scores.shape[-1],
            num_heads=attention_scores.shape[1],
            offset=offset,
            dtype=attention_scores.dtype,
        )
        if mask is not None:
            alibi_mask = alibi_mask + mask
        return attention_scores + alibi_mask
