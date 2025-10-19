import typing
from dataclasses import dataclass

import numpy as np


@dataclass
class Marker:
    # this type is kinda scuffed? but this is a hackathon
    tl: typing.Iterable[int]
    tr: typing.Iterable[int]
    br: typing.Iterable[int]
    bl: typing.Iterable[int]


@dataclass
class Box:
    tl: Marker
    tr: Marker
    bl: Marker
    br: Marker

    def inner_coordinates(self):
        return np.array((
            self.tl.br,
            self.tr.bl,
            self.br.tl,
            self.bl.tr,
        ), np.int32)

    def outer_coordinates(self):
        return np.array((
            self.tl.tl,
            self.tr.tr,
            self.br.br,
            self.bl.bl,
        ), np.int32)

