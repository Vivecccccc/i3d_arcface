import numpy as np
from data_process.detect_face import DetectFace
from data_process.seg_ins import SegmentBody
from utils.misc_utils import compute_bbox_size, compute_seg_size
from insightface.app.common import Face

class FaceRouter:
    def __init__(self, trace_face=False):
        self.frames = []
        self.valid_frames = []
        self.curr_frame = None
        self.main_face = []
        self.trace_face = trace_face
        self.face_detector = DetectFace()
        self.human_segmenter = SegmentBody()

    def record_frame(self, frame):
        self.frames.append(frame)
        self._decision()

    def _decision(self):
        loc_faces = self._has_face()
        is_valid, masked_frame, main_face = self._is_valid(loc_faces)
        if is_valid:
            self.valid_frames.append(masked_frame)
            self.main_face.append(main_face)

    def _has_face(self):
        self.curr_frame = self.frames[-1]
        loc_faces = self.face_detector.detect_faces(self.curr_frame)
        return loc_faces

    def _is_valid(self, loc_faces):
        bboxes, kpss = loc_faces
        flag = False
        human = None
        main_face = None
        if bboxes.shape[0] > 0:
            flag = True
            segment_id = 0
            humans = self.human_segmenter.make_segment(self.curr_frame, kpss)
            if bboxes.shape[0] > 1:
                segment_id = np.array(list(map(compute_seg_size, humans))).argmax()
            human = humans[segment_id]
            det_face = (bboxes[segment_id], kpss[segment_id])
            if self.trace_face:
                main_face = self.face_detector.create_face(det_face, self.curr_frame, True)
            else:
                main_face = self.face_detector.create_face(det_face)
        masked_frame = self.human_segmenter.blackout_background(self.curr_frame, human)
        return flag, masked_frame, main_face
        