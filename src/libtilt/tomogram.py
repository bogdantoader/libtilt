from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
import einops
from pydantic import BaseModel, validator

from libtilt.transformations import Rx, Ry, Rz, T, S
from libtilt.coordinate_utils import homogenise_coordinates
from libtilt.patch_extraction.subpixel_square_patch_extraction import extract_squares
from libtilt.rescaling import rescale_2d
from libtilt.backprojection import backproject_fourier
from libtilt.fft_utils import dft_center

TOMOGRAM_DIMENSIONS = (2000, 4000, 4000)


class VirtualTomogram(BaseModel):
    tilt_series: torch.Tensor  # (tilt, h, w)
    tilt_series_pixel_size: float
    # (d, h, w) at tilt-series pixel size
    tomogram_dimensions: tuple[int, int, int]
    eulers_xyz: torch.Tensor  # (tilt, 3) extrinsic XYZ rotation of tomogram
    shifts: torch.Tensor  # (tilt, dh, dw) in pixels
    target_pixel_size: float

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)

    def __hash__(self):  # weird, seems necessary for use of lru cache on method
        return id(self)

    @property
    def tomogram_center(self) -> torch.Tensor:
        # return torch.as_tensor(self.tomogram_dimensions, dtype=torch.float32) // 2
        return torch.as_tensor(TOMOGRAM_DIMENSIONS, dtype=torch.float32) // 2

    @property
    def tilt_image_center(self) -> torch.Tensor:  # (3, ) 0, center_h, center_w
        center = dft_center(
            image_shape=self.tilt_series.shape[-2:],
            rfft=False,
            fftshifted=True
        )
        return F.pad(center, (1, 0), value=0)

    @property
    def scale_factor(self) -> float:
        return self.tilt_series_pixel_size / self.target_pixel_size

    @property
    def rotation_matrices(self) -> torch.Tensor:
        r0 = Rx(self.eulers_xyz[:, 0], zyx=True)
        r1 = Ry(self.eulers_xyz[:, 1], zyx=True)
        r2 = Rz(self.eulers_xyz[:, 2], zyx=True)
        return (r2 @ r1 @ r0)[:, :3, :3]

    @property
    def transformation_matrices(self) -> torch.Tensor:
        t0 = T(-self.tomogram_center)
        s0 = S([self.scale_factor, self.scale_factor, self.scale_factor])
        r0 = Rx(self.eulers_xyz[:, 0], zyx=True)
        r1 = Ry(self.eulers_xyz[:, 1], zyx=True)
        r2 = Rz(self.eulers_xyz[:, 2], zyx=True)
        t1 = T(self.shifts_3d * self.scale_factor)
        t2 = T(self.tilt_image_center * self.scale_factor)
        return t2 @ t1 @ r2 @ r1 @ r0 @ s0 @ t0  # (tilt, 4, 4)

    @property
    def projection_matrices(self) -> torch.Tensor:
        return self.transformation_matrices[:, 1:3, :]  # (tilt, 2, 4)

    @property
    def shifts_3d(self) -> torch.Tensor:
        return F.pad(self.shifts, (1, 0), value=0)

    @lru_cache(maxsize=1)
    def rescale_tilt_series(self, target_pixel_size: float) -> torch.Tensor:
        tilt_series, spacing_rescaled = rescale_2d(
            self.tilt_series,
            source_spacing=self.tilt_series_pixel_size,
            target_spacing=target_pixel_size,
            maintain_center=True,
        )
        return tilt_series

    def calculate_projected_positions(
        self, particle_position: torch.Tensor
    ) -> torch.Tensor:
        particle_position = homogenise_coordinates(particle_position)
        particle_position = einops.rearrange(
            particle_position, 'zyxw -> zyxw 1')
        # (tilt, yx, 1)
        positions_2d = self.projection_matrices @ particle_position
        positions_2d = einops.rearrange(positions_2d, 'tilt yx 1 -> tilt yx')
        return positions_2d

    def extract_local_tilt_series(
        self, position_in_tomogram: torch.Tensor, sidelength: int
    ) -> torch.Tensor:  # (tilt, sidelength, sidelength)
        rescaled_tilt_series = self.rescale_tilt_series(
            target_pixel_size=self.target_pixel_size
        )
        projected_positions = self.calculate_projected_positions(
            position_in_tomogram)
        particle_tilt_series = extract_squares(
            image=rescaled_tilt_series,
            positions=projected_positions,
            sidelength=sidelength,
        )
        return particle_tilt_series

    def reconstruct_local_volume(
        self, position_in_tomogram: torch.Tensor, sidelength: int
    ) -> torch.Tensor:  # (sidelength, sidelength, sidelength)
        local_tilt_series = self.extract_local_tilt_series(
            position_in_tomogram=position_in_tomogram, sidelength=2 * sidelength
        )
        local_reconstruction = backproject_fourier(
            images=local_tilt_series,
            rotation_matrices=torch.linalg.inv(self.rotation_matrices),
            rotation_matrix_zyx=True,
        )
        low, high = sidelength // 2, (sidelength // 2) + sidelength
        return local_reconstruction[low:high, low:high, low:high]

    @validator('tilt_series', 'eulers_xyz', 'shifts', pre=True)
    def to_float32_tensor(cls, value):
        return torch.as_tensor(value).float()
