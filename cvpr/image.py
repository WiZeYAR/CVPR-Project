from cvpr.frame import Frame


class Image(Frame):
    def __init__(self, filename: str) -> None:
        from cv2 import imread

        super().__init__(imread(filename))
