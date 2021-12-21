#!/usr/bin/env python3
import cv2
import numpy as np
from playsound import playsound
import time
from threading import Thread
import terminator as t


#
# Camera, HUD overlays, sounds...
#
T800_CAMERA = 0
HUD_VIDEO = "overlay2.mp4"
HUD_SOUND = "overlay2.mp3"
WINDOW_NAME = "T800 Vision"

ATMOSPHERIC_SOUND = "overlay1.mp3"
SCAN_TIMINGS = [6499, 7143, 7985, 9079, 9996, 10908, 11993, 29156, 32874, 36394,
                40458, 42278, 47110, 48779, 53049, 55266, 57667, 84541, 88201,
                91745, 95779, 97582, 102441, 104121, 108402, 110597, 112997,
                999999]
SCAN_DURATION = 200


#
# Play atmosphere background on loop...
#
def play_atmosphere(name):
    while True:
        playsound(name)


playThread = Thread(target=play_atmosphere, args=(
    ATMOSPHERIC_SOUND,), daemon=True)
playThread.start()


#
# Overlay HUD and scans on visual input...
#
hud = t.HeadsUpDisplay(HUD_VIDEO, HUD_SOUND)  # Access HUD analysis...
cam = t.TerminatorVision(T800_CAMERA, hud.get_shape())  # Access T800 visual cortex...
scanIndex = 0  # Index to next SCAN_TIMINGS entry.
scanned = False  # Indicates we have scanned within the SCAN_TIMING window.

while True:
    camFrame = cam.read()
    hudFrame = hud.read()

    # Loop scan timings.
    timeMillis = hud.get_time()
    if scanIndex > 0 and timeMillis <= SCAN_TIMINGS[0]:
        scanIndex = 0
        scanned = False

    # In a scan window, so edge scan.
    if SCAN_TIMINGS[scanIndex] <= timeMillis < SCAN_TIMINGS[scanIndex] + SCAN_DURATION:
        # Only scan once (so it does not shimmer).
        if not scanned:
            # Canny Edge Detection.
            edges = cv2.Canny(image=camFrame, threshold1=100, threshold2=200)
            scanFrame = cv2.merge([edges, edges, edges])
            scanned = True
    elif timeMillis >= SCAN_TIMINGS[scanIndex] + SCAN_DURATION:
        scanIndex += 1  # Ready for next scan timing.
        scanned = False

    if scanned:  # Overlay HUD and scan on camera input.
        out = cv2.addWeighted(cv2.addWeighted(
            camFrame, 1.0, scanFrame, 1.0, 0), 1.0, hudFrame, 1.0, 0)
    else:  # No scan - just HUD overlay on camera input.
        out = cv2.addWeighted(camFrame, 1.0, hudFrame, 1.0, 0)

    #
    # Display the resulting frame...
    #
    cv2.imshow(WINDOW_NAME, out)

    if cv2.waitKey(1) == 27:
        break

#
# When everything done, release the capture
#
cam.release()
hud.release()
cv2.destroyAllWindows()
