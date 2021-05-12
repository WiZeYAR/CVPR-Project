#!/usr/bin/env python

###
### THIS PROGRAM FINDS A GAME TABLE BASED ON
### IMAGE FLUX (WHICH SHOWS EDGES), AND
### JACCARD SIMILARITY (WHICH CHECKS FOR DISTANCE BETWEEN IMAGES)
###
### USAGE: python find_game_table.py [out_file]
###

from cvpr.video import Video
from typing import Tuple
from cvpr.image import Image
from cvpr.frame import Frame


def crop_table_borders(
    img: Frame, denoise_param: Tuple[float, float, float] = (0, 1 / 20, 0)
) -> Tuple[Frame, Frame, Frame, Frame]:
    left, right, top, bot = map(
        lambda frame: frame.edges(denoise_param),
        map(
            lambda s: img.crop(s),
            [
                (slice(None, None), slice(150, 400), slice(None, None)),
                (slice(None, None), slice(850, 1100), slice(None, None)),
                (slice(20, 55), slice(150, 1100), slice(None, None)),
                (
                    slice(600, 701),
                    slice(150, 1100),
                    slice(None, None),
                ),
            ],
        ),
    )
    return left, right, top, bot


def is_similar(
    a: Tuple[Frame, Frame, Frame, Frame],
    b: Tuple[Frame, Frame, Frame, Frame],
    tol: float = 0.2,
) -> bool:
    return min(map(lambda p: p[0].diff(p[1]), zip(a, b))) <= tol


if __name__ == "__main__":
    from sys import argv
    from tqdm import tqdm

    print("Scanning video file...")
    example = crop_table_borders(Image("WSC sample good.png"))
    with open(argv[1], "w") as file:
        for i, frame in tqdm(enumerate(Video("WSC 2.mp4", fps=30))):
            file.write(
                f"{i} {min(map(lambda p: p[0].diff(p[1]), zip(example, crop_table_borders(frame))))}\n"
            )
    print("Job done")
