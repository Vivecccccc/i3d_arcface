import os
import cv2
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_process.face_router import FaceRouter


def face_focus_transform(frame, face):
    import cv2
    from insightface.utils.face_align import norm_crop
    img = norm_crop(frame, face.kps)
    img = cv2.dnn.blobFromImage(img, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    return img

def transform(frames):
    # Normalize pixel values to [0, 1]
    frames = frames / 255.0
    # Resize frames to (224, 224)
    frames = cv2.resize(frames, (224, 224))
    # Convert frames to PyTorch tensor
    frames = torch.from_numpy(frames).float()
    # Add a batch dimension
    frames = frames.unsqueeze(0)
    return frames

class VideoDataset(Dataset):
    def __init__(self, data_path, save_root, max_frames, 
                 transform=face_focus_transform, 
                 sample_fps=25, 
                 trace_face=False,
                 recorded=False):
        self.data_path = data_path
        self.save_root = save_root
        self.transform = transform
        self.max_frames = max_frames
        self.labels = []
        self.trace_face = trace_face
        self.filenames = []

        if recorded:
            with open(os.path.join(save_root, 'labels.txt'), 'r') as f:
                _ = f.read()
            self.labels = [int(x) for x in _.split('\n')]
            with open(os.path.join(save_root, 'filenames.txt'), 'r') as f:
                _ = f.read()
            self.filenames = [x for x in _.split('\n')]
        else:
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            for filename in tqdm(os.listdir(data_path)):
                if not filename.endswith('.mp4'):
                    continue
                router = FaceRouter(trace_face=self.trace_face)
                cap = cv2.VideoCapture(os.path.join(data_path, filename))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(round(fps / sample_fps)) if fps > sample_fps else 1
                i = 0
                while True:
                    i += 1
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if i % frame_interval == 0:
                        router.record_frame(frame)
                valid_frames = router.valid_frames
                valid_faces = router.main_face
                frame_count = len(valid_frames)
                assert len(valid_faces) == frame_count
                if frame_count < max_frames:
                    continue
                valid_frames_lst = [valid_frames[i: i+max_frames] for i in range(0, frame_count, max_frames) if i+max_frames <= frame_count]
                valid_faces_lst = [valid_faces[i: i+max_frames] for i in range(0, frame_count, max_frames) if i+max_frames <= frame_count]
                curr_idx = len(self.labels)
                label = int(filename.split('_')[0])
                for idx, (valid_frames, valid_faces) in enumerate(zip(valid_frames_lst, valid_faces_lst)):
                    self._save_frames(valid_frames, idx + curr_idx)
                    self._save_faces(valid_faces, idx + curr_idx)
                    self.labels.append(label)
                    self.filenames.append(filename)
                del router
                del valid_frames_lst
                del valid_faces_lst
            with open(os.path.join(save_root, 'labels.txt'), 'w') as f:
                f.write('\n'.join([str(x) for x in self.labels]))
            with open(os.path.join(save_root, 'filenames.txt'), 'w') as f:
                f.write('\n'.join([str(x) for x in self.filenames]))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        valid_frames = self._load_frames(idx)
        valid_faces = self._load_faces(idx)
        label = self.labels[idx]
        filename = self.filenames[idx]
        assert len(valid_frames) == len(valid_faces) == self.max_frames
        rgb_frames, motion_frames = [], []
        for i, (frame, face) in enumerate(zip(valid_frames, valid_faces)):
            if self.trace_face and i == 0:
                prev_face_ldmk = face['landmark_2d_106']
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame_rgb = self.transform(frame_rgb, face)
            rgb_frames.append(frame_rgb)
            if self.trace_face:
                curr_face_ldmk = face['landmark_2d_106']
                # TODO
        rgb_frames = np.concatenate(rgb_frames)
        rgb_frames = np.transpose(rgb_frames, (1, 0, 2, 3))
        return rgb_frames, label, filename
    
    def _save_frames(self, lst_of_frame_arr, idx):
        filename = os.path.join(self.save_root, f'x_{idx}.npz')
        np.savez_compressed(filename, *lst_of_frame_arr)

    def _load_frames(self, idx):
        filename = os.path.join(self.save_root, f'x_{idx}.npz')
        max_frames = self.max_frames
        loaded = np.load(filename)
        arr_name = (f'arr_{i}' for i in range(max_frames))
        lst = []
        for x in arr_name:
            lst.append(loaded[x])
        return lst
    
    def _save_faces(self, lst_of_face_inst, idx):
        import pickle
        def _save_face(face, trace_face):
            D = {'kps': face['kps'], 'bbox': face['bbox'], 'det_score': face['det_score']}
            if trace_face:
                D.update({'landmark_2d_106': face['landmark_2d_106']})
            return D    
        filename = os.path.join(self.save_root, f'f_{idx}.pkl')
        _lst = list(map(lambda x: _save_face(x, self.trace_face), lst_of_face_inst))
        with open(filename, 'wb') as f:
            pickle.dump(_lst, f)
    
    def _load_faces(self, idx):
        import pickle
        filename = os.path.join(self.save_root, f'f_{idx}.pkl')
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
        from insightface.app.common import Face
        _lst = [Face(elem) for elem in loaded]
        return _lst

def main(args):
    data_path = args.data_path
    save_root = args.save_root
    max_frames = args.max_frames
    V = VideoDataset(data_path=data_path, save_root=save_root, max_frames=max_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path of the videos")
    parser.add_argument("--save_root", type=str, help="path to save the processed data")
    parser.add_argument("--max_frames", type=int)
    args = parser.parse_args()

    main(args)