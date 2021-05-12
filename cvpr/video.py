from __future__ import annotations
from typing import Generator, Iterator
from cvpr.frame import Frame


class Video:
    def __init__(self, filename: str, fps: float) -> None:
        from cv2 import VideoCapture

        self.__filename = filename
        self.__cap = VideoCapture(filename)
        self.__fps = fps

    def __iter__(self) -> Iterator[VideoFrame]:
        return VideoStream(self)

    def __getitem__(self, frame_no: int) -> Frame:
        self.__cap.set(0, round(frame_no * 1000 / self.fps))
        return VideoFrame(self.__cap.read()[1])

    @property
    def filename(self) -> str:
        return self.__filename

    @property
    def fps(self) -> float:
        return self.__fps


class VideoFrame(Frame):
    ...


def VideoStream(video: Video) -> Generator[VideoFrame, None, None]:
    from cv2 import VideoCapture
    from numpy import ndarray

    success: bool
    img: ndarray

    cap = VideoCapture(video.filename)
    success, img = cap.read()
    while success:
        yield VideoFrame(img)
        success, img = cap.read()
