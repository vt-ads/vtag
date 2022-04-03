"""
VTSream: a file stream like object to manage the input media
"""
import os
import numpy as np
import cv2 as cv

class VTStream():
    def __init__(self):
        self.filetype = None # mp4, jpg, png
        self.files    = None
        self.dirname  = None
        self.dirsave  = None
        self.stream   = None
        self.isEmpty  = True
        self.n = 0
        self.h = 0
        self.w = 0
        self.h_limit = 480

    def load(self, path):
        self.files, self.dirname, self.filetype = get_files(path)
        # check if the folder exist
        self.dirsave = os.path.join(self.dirname, "vtag")
        if not os.path.exists(self.dirsave):
            os.makedirs(self.dirsave)

    def get_meta(self):
        if self.filetype == "mp4":
            if self.stream is not None: self.stream.release()
            self.stream = cv.VideoCapture(self.files)
            n = int(self.stream.get(cv.CAP_PROP_FRAME_COUNT))
            w = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        else:
            n    = len(self.files)
            img  = cv.imread(self.files[0], cv.IMREAD_GRAYSCALE)
            h, w = img.shape
        # resize video, image
        if h > self.h_limit:
            scale_ratio = self.h_limit / h
            h = int(h * scale_ratio)
            w = int(w * scale_ratio)
        # update parameters
        self.n, self.h, self.w = n, h, w
        self.isEmpty = False if self.n > 0 else True
        # return
        return n, h, w

    def get(self, i_start, i_end=None):
        """
        get frames
        """
        n =  1 if i_end is None else i_end - i_start
        frames = np.zeros((n, self.h, self.w), dtype=np.uint8)
        if self.filetype == "mp4":
            self.stream.set(cv.CAP_PROP_POS_FRAMES, i_start)
            for i in range(n):
                _, frame_read = self.stream.read()
                frame_read = cv.cvtColor(frame_read, cv.COLOR_BGR2GRAY)
                frames[i, :, :] = cv.resize(frame_read, (self.w, self.h))
                # cv.waitKey(1)
        else:
            for i in range(n):
                frame_read = cv.imread(self.files[i_start + i],
                                       cv.IMREAD_GRAYSCALE)
                frames[i, :, :] = cv.resize(frame_read, (self.w, self.h))

        if n == 1:
            return frames.reshape((self.h, self.w))
        else:
            return frames

def get_files(path):
    """
    inputs
    ---
    path: path to the files of interest

    outputs
    ---
    files: list of paths to the files of interest (could be one or many)
    dirname: the directory containing the files
    filetype: extension name

    example
    ---
    path = os.path.join(root, "data", "pig_3_sample")

    """

    is_dir = os.path.isdir(path)
    if path.split(".")[-1] == "mp4":
        # video
        files    = path
        filetype = "mp4"
        dirname = os.path.dirname(path)
    else:
        # if images, get the directory path
        if not is_dir:
            dirname = os.path.dirname(path)
        else:
            dirname = path
        # get absolute path to the files
        files = [os.path.join(dirname, f) for f in os.listdir(dirname)]
        # get most frequent extension name
        extension = np.array([f.split(".")[-1] for f in files])
        name, counts = np.unique(extension, return_counts=True)
        filetype = name[np.argmax(counts)]
        files = [f for f in files if filetype in f]
        files.sort()

    return files, dirname, filetype
