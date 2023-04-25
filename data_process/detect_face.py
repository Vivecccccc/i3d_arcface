from insightface.app import FaceAnalysis
from insightface.app.common import Face

class DetectFace:
    def __init__(self, cuda=True, det_size=(640, 640)):
        exec_providers = {True: 'CUDAExecutionProvider', 
                          False: 'CPUExecutionProvider'}
        self.detector = FaceAnalysis(providers=[exec_providers[cuda]])
        self.detector.prepare(ctx_id=0, det_size=det_size)

    def detect_faces(self, img):
        bboxes, kpss = self.detector.det_model.detect(img)
        return bboxes, kpss
    
    def create_face(self, det_face, img=None, landmark=False):
        bbox, kps = det_face
        _ = bbox[4]
        bbox = bbox[:4]
        _face = Face(bbox=bbox, kps=kps, det_score=_)
        if img is None:
            return _face
        if landmark:
            landmark_model = self.detector.models['landmark_2d_106']
            landmark_model.get(img, _face)
        return _face
        