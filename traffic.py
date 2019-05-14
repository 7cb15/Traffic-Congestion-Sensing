import os
import sys
import logging
import logging.handlers
import random

import numpy as np
# import skvideo.io
import cv2
# import matplotlib.pyplot as plt

import utils
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
random.seed(123)


from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

# ============================================================================
IMAGE_DIR = sys.argv[1]
VIDEO_SOURCE = sys.argv[2]
REPORT_NAME = sys.argv[3]
# IMAGE_DIR = "../out_test"
# VIDEO_SOURCE = "../videos/sample.mp4"
# IMAGE_FOLDER = "../gopro_cropped"
SHAPE = (1124, 1500)  # HxW
EXIT_PTS = np.array([
    # morning
    [[680, 600], [660, 525], [870, 490], [920, 570]]
    # [[900,1200],[850,1060],[1230,1000],[1320,1140]] #sample
])
# FILTER_MASK = np.array([
#             [[840, 1125], [590, 290], [690, 290], [1280, 1125]]
#             # [[940, 1460], [620, 400], [800, 390], [1425, 1420]] # sample
#         ])
# ============================================================================


def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print('Training BG Subtractor...')
    i = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        #     for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    # there is also bgslibrary, that seems to give better BG substruction, but
    # not tested it yet
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    # processing pipeline for programming conviniance
    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor, save_image=True, image_dir=IMAGE_DIR),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0, x_weight=1.5),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name=REPORT_NAME)
    ], log_level=logging.DEBUG)

    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # skipping 100 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=500)

    frame_number = -1
    while True:
        ret, frame = cap.read()
        if frame is None:
            log.error("Frame capture failed, stopping...")
            break

        frame_number += 1

        # plt.imshow(frame)
        # plt.show()
        # return

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()

# ============================================================================

if __name__ == "__main__":
    '''
    arg 1: IMAGE_DIR = path to image output. eg "../out_test"
    arg 2: VIDEO_SOURCE = path to video file. eg "../videos/sample.mp4"
    arg 3: REPORT_NAME = path to csv report. eg "report.csv"
    '''
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
