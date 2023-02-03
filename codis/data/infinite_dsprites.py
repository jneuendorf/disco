import os
from collections import namedtuple
from itertools import count, islice, product
from typing import Iterable

import imageio
import numpy as np
import pygame
import pygame.gfxdraw
from scipy.interpolate import splev, splprep
from torch.utils.data import IterableDataset

BaseLatents = namedtuple(
    "BaseLatents", "color shape scale orientation position_x, position_y"
)


class Latents(BaseLatents):
    """Latent variables defining a single image."""

    def to(self, device):
        return Latents(
            self.color.to(device),
            self.shape.to(device),
            self.scale.to(device),
            self.orientation.to(device),
            self.position_x.to(device),
            self.position_y.to(device),
        )

    def __getitem__(self, key):
        return getattr(self, key)


class InfiniteDSprites(IterableDataset):
    def __init__(
        self,
        image_size: int = 64,
        scale_range: Iterable = np.linspace(0.5, 1, 6),
        orientation_range: Iterable = np.linspace(0, 2 * np.pi, 40),
        position_x_range: Iterable = np.linspace(0, 1, 32),
        position_y_range: Iterable = np.linspace(0, 1, 32),
        min_verts: int = 3,
        max_verts: int = 10,
        radius_std: float = 0.6,
        angle_std: float = 0.8,
    ):
        """Create a dataset of images of random shapes.
        Args:
            image_size: The size of the images in pixels.
            scale_range: The range of scales to use.
            orientation_range: The range of orientations to use.
            position_x_range: The range of x positions to use.
            position_y_range: The range of y positions to use.
            min_verts: The minimum number of vertices in the shape.
            max_verts: The maximum number of vertices in the shape.
            radius_std: The standard deviation of the radius of the vertices.
            angle_std: The standard deviation of the angle of the vertices.
        Returns:
            None
        """
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        self.image_size = image_size
        self.ranges = {
            "shape": (self.generate_shape() for _ in count()),
            "scale": scale_range,
            "orientation": orientation_range,
            "position_x": position_x_range,
            "position_y": position_y_range,
        }
        self.min_verts = min_verts
        self.max_verts = max_verts
        self.radius_std = radius_std
        self.angle_std = angle_std
        self.window = pygame.display.set_mode((self.image_size, self.image_size))

    def __iter__(self):
        """Generate an infinite stream of images and latent vectors.
        Args:
            None
        Returns:
            An infinite stream of (image, latents) tuples."""
        color = 0
        while True:
            for shape, scale, orientation, position_x, position_y in product(
                *self.ranges.values()
            ):
                latents = Latents(
                    color, shape, scale, orientation, position_x, position_y
                )
                image = self.draw(latents)
                yield image, latents

    def generate_shape(self):
        """Generate random vertices and connect them with straight lines or a smooth curve.
        Args:
            min_verts: Minimum number of vertices (inclusive).
            max_verts: Maximum number of vertices (inclusive).
            radius_std: Standard deviation of the polar radius when sampling the vertices.
            angle_std: Standard deviation of the polar angle when sampling the vertices.
        Returns:
            A tuple of (points, spline), where points is an array of points of shape (2, num_verts)
            and spline is an array of shape (2, num_spline_points).
        """
        verts = self.sample_vertex_positions()
        shape = (
            self.interpolate(verts)
            if np.random.rand() < 0.5
            else self.interpolate(verts, k=1)
        )
        shape = shape / np.max(np.linalg.norm(shape, axis=0))  # normalize scale
        shape = shape - np.mean(shape, axis=1, keepdims=True)  # center shape
        return shape

    def sample_vertex_positions(self):
        """Sample the positions of the vertices of a polygon.
        Args:
            radius_std: Standard deviation of the polar radius when sampling the vertices.
            angle_std: Standard deviation of the polar angle when sampling the vertices.
            num_verts: Number of vertices.
        Returns:
            An array of shape (2, num_verts).
        """
        num_verts = np.random.randint(self.min_verts, self.max_verts + 1)
        rs = np.random.normal(1.0, self.radius_std, num_verts)
        rs = np.clip(rs, 0.1, 1.9)

        epsilon = 1e-6
        circle_sector = np.pi / num_verts - epsilon
        intervals = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
        thetas = np.random.normal(0.0, circle_sector * self.angle_std, num_verts)
        thetas = np.clip(thetas, -circle_sector, circle_sector) + intervals

        verts = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(rs, thetas)]
        verts = np.array(verts).T
        return verts

    @staticmethod
    def interpolate(verts: np.array, k: int = 3, num_spline_points: int = 1000):
        """Interpolate a set of vertices with a spline.
        Args:
            verts: An array of shape (2, num_verts).
            k: The degree of the spline.
            num_spline_points: The number of points to sample from the spline.
        Returns:
            An array of shape (2, num_spline_points).
        """
        verts = np.column_stack((verts, verts[:, 0]))
        spline_params, u = splprep(verts, s=0, per=1, k=k)
        u_new = np.linspace(u.min(), u.max(), num_spline_points)
        x, y = splev(u_new, spline_params, der=0)
        return np.array([x, y])

    def draw(self, latents: Latents):
        """Draw an image based on the values of the latents.
        Args:
            window: The pygame window to draw on.
            latents: The latents to use for drawing.
        Returns:
            The image as a numpy array.
        """
        self.window.fill(pygame.Color("black"))
        shape = self.apply_scale(latents.shape, latents.scale)
        shape = self.apply_orientation(shape, latents.orientation)
        shape = self.apply_position(shape, latents.position_x, latents.position_y)
        pygame.gfxdraw.aapolygon(self.window, shape.T.tolist(), (255, 255, 255))
        pygame.draw.polygon(self.window, pygame.Color("white"), shape.T.tolist())
        pygame.display.update()
        return pygame.surfarray.array3d(self.window)

    def apply_scale(self, shape, scale):
        """Apply a scale to a shape."""
        return 0.2 * self.image_size * scale * shape

    @staticmethod
    def apply_orientation(shape, orientation):
        """Apply an orientation to a shape.
        Args:
            shape: An array of shape (2, num_points).
            orientation: The orientation in radians.
        Returns:
            The rotated shape.
        """
        rotation_matrix = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )
        return rotation_matrix @ shape

    def apply_position(self, shape, position_x, position_y):
        """Apply a position to a shape.
        Args:
            shape: An array of shape (2, num_points).
            position_x: The x position of the shape.
            position_y: The y position of the shape.
        Returns:
            An array of shape (2, num_points).
        """
        height, width = self.window.get_size()
        position = np.array(
            [
                0.25 * height + 0.5 * height * position_y,
                0.25 * width + 0.5 * width * position_x,
            ]
        ).reshape(2, 1)
        return shape + position

    def sample_latents(self):
        """Sample a random set of latents."""
        return Latents(
            color=0,
            shape=self.generate_shape(),
            scale=np.random.choice(self.ranges["scale"]),
            orientation=np.random.choice(self.ranges["orientation"]),
            position_x=np.random.choice(self.ranges["position_x"]),
            position_y=np.random.choice(self.ranges["position_y"]),
        )


class InfiniteDSpritesTriplets(InfiniteDSprites):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Generate a tuple containing a triplet of images and an action encoding.
        See the appendix of Montero et al. (2022) for details.
        Args:
            None
        Yields:
            A tuple of ((image_original, image_transform, image_target), action).
        """
        while True:
            action = np.random.choice(list(self.ranges.keys()))
            latents_original = self.sample_latents()
            latents_transform = self.sample_latents()
            if (
                action != "shape"
                and latents_original[action] == latents_transform[action]
            ):
                continue
            latents_target = latents_original._replace(
                **{action: latents_transform[action]}
            )
            image_original = self.draw(latents_original)
            image_transform = self.draw(latents_transform)
            image_target = self.draw(latents_target)
            yield ((image_original, image_transform, image_target), action)


if __name__ == "__main__":
    dataset = InfiniteDSprites(image_size=512)
    writer = imageio.get_writer("infinite_dsprites.gif", mode="I")
    for image, _ in islice(dataset, 1000):
        writer.append_data(image)
    writer.close()