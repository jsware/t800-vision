"""
Cyberdyne Systems Series 800 Model 101 Infiltration Unit - Terminator T800.

Logic for the CSM101 T800 infiltration unit. Provides access to visual cortex
and Heads-Up-Display (HUD) analysis.

  * TerminatorVision: The T800 Visual cortex.
  * HeadUpDisplay: The HUD analysis.
"""
import cv2  # OpenCV module.
import numpy as np
from playsound import playsound
import time
from threading import Thread


class TerminatorVision:
    """
    CSM 101 Terminator T800 Visual Input.

    Receive and process data from Terminator's visual cortext. Input is received
    and processed asynchronously for improved performance.

    Throws exceptions if errors occur.
    """

    def __init__(self, feed: int, shape: tuple):
        """ Construct Terminator visual input feed. """
        self.feed: int = feed
        self.shape = tuple(int(x) for x in shape)

        # Open camera.
        self.cam = cv2.VideoCapture(self.feed)
        if not self.cam.isOpened():
            raise Exception(f"Accessing camera #{self.feed} error.")

        # Work out processing metrics from sample video feed frame.
        (self.ret, self.frame) = self.cam.read()
        if not self.ret:
            raise Exception(
                f"Reading initial frame from camera #{self.feed} error.")

        # Create empty channel to merge with red channel.
        self.ZEROS = np.zeros(self.frame[:, :, 2].shape, dtype="uint8")

        # Capture the first image to ensure we have one before thread starts.
        self.__capture()

        # Start background thread to read camera asynchronously (better performance).
        self.stopping = False
        Thread(target=self.__update, args=(), daemon = True).start()

    def __update(self):
        """
        Update with the latest camera image.

        Runs in the background on a separate thread for additional performance.
        """
        while not self.stopping and self.ret:
            self.__capture()

        self.cam.release()

    def __capture(self):
        """ Capture and process a frame from visual input feed. """
        (camRet, camFrame) = self.cam.read()
        if camRet:
            camFrame = cv2.merge([self.ZEROS, self.ZEROS, camFrame[:, :, 2]])
            if camFrame.shape[:2] != self.shape[:2]:
                camFrame = self.__resize(camFrame)
            
        (self.ret, self.frame) = (camRet, camFrame)

    def __resize(self, image):
        """ Resize (and letterbox) the image to the required dimensions. """
        (reqH, reqW) = self.shape[:2]
        (imgH, imgW) = image.shape[:2]

        ratio = reqW / float(imgW)
        dim = (reqW, int(imgH * ratio))

        # Resize the image
        resized = cv2.resize(image, dim)
        (imgH, imgW) = resized.shape[:2]

        # Letterbox the image
        y = (imgH - reqH) / 2

        return resized[int(y):int(y+reqH), 0:reqW]

    def read(self):
        """
        Return the last processed image from visual input feed.
        """
        if not self.ret:
            raise Exception(f"Unable to read frame from camera {self.feed}")
        else:
            return self.frame

    def release(self):
        """ Release the visual cortex camera as part of a shutdown process. """
        self.stopping = True

    def __del__(self):
        """ Destroy the T800 vision object and release the camera. """
        self.release()


class HeadsUpDisplay:
    """
    CSM 101 Terminator T800 Heads-Up Display (HUD).

    Receive and process HUD analysis data from T800 Neural Net Processor.
    """

    def __init__(self, feed: str, sound: str):
        """ Construct T800 HUD Feed. """
        self.feed = feed
        self.sound = sound

        self.hud = cv2.VideoCapture(feed)
        if not self.hud.isOpened():
            raise Exception(f"Accessing HUD analysis feed {self.feed} error.")

        self.FPS = self.hud.get(cv2.CAP_PROP_FPS)
        self.frameNumber = 0 # Next frame number of HUD.
        self.baseNanos = None # Base time for relative time calculations (None => set when read)
        self.timeMillis = 0 # Relative time of last frame read.

    def read(self):
        """
        Return correct frame of analysis from HUD.

        Will return the current frame of analysis or the next frame of analysis. Will
        skip ahead to keep analysis feed in-sync.

        Returns: A frame of analysis to match the current time.
        """
        if self.baseNanos is None:
            playsound(self.sound, block=False)
            # Base for working out relative time in milliseconds.
            self.baseNanos = time.time_ns()

        # Handle delays to keep video & sound in step.
        # Time relative to beginning of loop.
        self.timeMillis = (time.time_ns() - self.baseNanos) // 1000000

        # Calc frame required from relative time and FPS.
        frameRequired = self.timeMillis * self.FPS // 1000

        # Skip to sync any lag.
        if self.frameNumber < frameRequired:
            self.__skip(frameRequired)

        if self.frameNumber == frameRequired:
            # Read HUD frame if we are ready for it.
            (self.ret, self.frame) = self.hud.read()

            if self.ret:
                self.frameNumber += 1
            elif self.frameNumber == 0 and not self.ret:
                raise Exception(
                    f"Failed to read initial frame {self.frameNumber} from {self.src}.")
            else:  # If not read frame 1+, loop and try again.
                self.__loop()
                self.read()

        return self.frame


    def __skip(self, frameRequired):
        """ Skip frames to catch up on lag. """
        print(f"@{self.timeMillis}: skip {self.frameNumber} to {frameRequired}")

        if not self.hud.set(cv2.CAP_PROP_POS_FRAMES, frameRequired):
            raise Exception(
                f"Failed to skip from {self.frameNumber} to {frameRequired}.")

        self.frameNumber = frameRequired


    def __loop(self):
        """ Loop the HUD feed. """
        self.frameNumber = 0
        if not self.hud.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber):
            raise Exception(f"Failed to loop to {self.frameNumber}.")
        self.baseNanos = None # Force read to restart time.
        self.timeMillis = 0

    def get_time(self):
        """ Return relative time in ms since start of HUD feed. """
        return self.timeMillis

    def get_shape(self):
        """ Return shape (width, height) of HUD frames. """
        return (int(self.hud.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.hud.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def release(self):
        """ Release the HUD analysis feed as part of a shutdown process. """
        self.stopping = True

    def __del__(self):
        """ Destroy the T800 HUD object and release the analysis feed. """
        self.release()
