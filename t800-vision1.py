#!/usr/bin/env python3
import cv2
import terminator as t


#
# Camera, HUD overlays, sounds...
#
T800_CAMERA = 0
HUD_VIDEO = "overlay1.mp4"
HUD_SOUND = "overlay1.mp3"
WINDOW_NAME = "T800 Vision"


#
# Overlay HUD on visual input...
#
cam = t.TerminatorVision(T800_CAMERA)  # Access T800 visual cortex...
hud = t.HeadsUpDisplay(HUD_VIDEO, HUD_SOUND)  # Access HUD analysis...
while True:
    camFrame = cam.read()
    hudFrame = hud.read()
    out = cv2.addWeighted(camFrame, 1.0, hudFrame, 1.0, 0)  # Overlay HUD.

    # Display the resulting frame...
    cv2.imshow(WINDOW_NAME, out)

    if cv2.waitKey(1) == 27:
        break


#
# When everything done, release visual input and HUD...
#
cam.release()
hud.release()
cv2.destroyAllWindows()
