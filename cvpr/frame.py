from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union


class Frame:
    def __init__(self, data: np.ndarray) -> None:

        # ---- STATE ---- #
        self.__data: np.ndarray = data
        self.__hough_transform: Optional[Frame] = None
        self.__edges: Optional[Frame] = None

        # ---- STUFF ---- #
        self.__data.flags.writeable = False

    def __getitem__(self, idx) -> np.ndarray:
        height, width, channel = idx
        return self.__data[height, width, channel]

    def show(self) -> None:
        from matplotlib.pyplot import imshow, show
        from cv2 import cvtColor, COLOR_BGR2RGB

        imshow(cvtColor(self.__data, COLOR_BGR2RGB))
        show()

    def edges(
        self, channel_multiplier: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    ) -> Frame:
        from cv2 import Sobel, CV_32F
        from numpy import abs, array, round

        # ---- LAZY EVALUATION ---- #
        if self.__edges is not None:
            return self.__edges

        # ---- FIRST EVALUATION ---- #
        self.__edges = Frame(
            round(
                array(
                    sum(
                        map(
                            lambda i: channel_multiplier[i]
                            / 2
                            * abs(
                                Sobel(self[:, :, i], CV_32F, 0, 1)
                                + channel_multiplier[i]
                                / 2
                                * abs(Sobel(self[:, :, i], CV_32F, 1, 0)),
                            ),
                            range(3),
                        )
                    )
                )
            )
        )

        return self.edges()

    def diff(self, other: Frame) -> float:
        from scipy.spatial.distance import jaccard
        from numpy import sum
        from functools import reduce
        from operator import mul

        assert self.__data.shape == other.__data.shape

        return jaccard(self.__data.reshape((-1,)), other.__data.reshape((-1,)))

        # self.__data.reshape((-1,))
        # return sum(self.__data != other.__data) / reduce(mul, self.__data.shape)

    def crop(self, idx: Tuple[slice, slice, slice]) -> Frame:
        width, height, channel = idx
        return Frame(self.__data[width, height, channel])
