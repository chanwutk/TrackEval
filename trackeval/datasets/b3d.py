import json

import numpy as np

from ._base_dataset import _BaseDataset
from .. import _timing


LIMIT = 5000


class B3D(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        return {}

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        self.tracker_list = ['SORT']
        self.seq_list = ['']
        self.class_list = ['car']
        self.output_fol, self.output_sub_fol = 'output-eval', 'not_interpolated_tracks'

    def get_display_name(self, tracker):
        return 'SORT'

    def _load_raw_file(self, tracker, seq: str, is_gt: bool):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        data = {}
        if is_gt:
            file = 'tracks.jsonl'
            with open(file, 'r') as f:
                lines = f.readlines()
            
            data['num_timesteps'] = len(lines)
            data['gt_ids'] = []
            data['gt_classes'] = []
            data['gt_dets'] = []
            data['gt_crowd_ignore_regions'] = []
            data['gt_extras'] = []

            for idx, line in enumerate(lines):
                if idx > LIMIT:
                    break
                line = json.loads(line)
                assert idx == int(line[0])
                gt_ids = []
                gt_classes = []
                gt_dets = []
                gt_extras = []
                for det in line[1]:
                    gt_ids.append(det[0])
                    gt_classes.append(0)
                    gt_dets.append(det[1:])
                    gt_extras.append({})
                data['gt_ids'].append(np.array(gt_ids, dtype=int))
                data['gt_classes'].append(np.array(gt_classes, dtype=int))
                data['gt_dets'].append(np.array(gt_dets))
                data['gt_crowd_ignore_regions'].append([])
                data['gt_extras'].append(gt_extras)
            
            data['seq'] = seq
            return data
        else:
            file = self.output_sub_fol + '.jsonl'
            with open(file, 'r') as f:
                trajectories = f.readlines()
            
            tracks = {}
            for tid, t in enumerate(trajectories):
                _, trajectory = json.loads(t)
                for det in trajectory:
                    frame = det[0]
                    if frame > LIMIT:
                        continue
                    box = det[1]
                    if frame not in tracks:
                        tracks[frame] = []
                    tracks[frame].append([tid, box])
            
            data['num_timesteps'] = max(tracks.keys()) + 1
            data['tracker_ids'] = []
            data['tracker_classes'] = []
            data['tracker_dets'] = []
            data['tracker_confidences'] = []

            for t in range(data['num_timesteps']):
                if t in tracks:
                    data['tracker_ids'].append(np.array([det[0] for det in tracks[t]], dtype=int))
                    data['tracker_dets'].append(np.array([det[1] for det in tracks[t]]))
                    data['tracker_classes'].append(np.array([0 for _ in range(len(tracks[t]))], dtype=int))
                    data['tracker_confidences'].append(np.array([1 for _ in range(len(tracks[t]))]))
                else:
                    data['tracker_ids'].append(np.zeros((0,), dtype=int))
                    data['tracker_dets'].append(np.zeros((0, 4)))
                    data['tracker_classes'].append(np.zeros((0,), dtype=int))
                    data['tracker_confidences'].append(np.ones((0,)))
            
            data['seq'] = seq
            return data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        num_tracker_ids = set()
        for ids in raw_data['tracker_ids']:
            num_tracker_ids.update(ids)

        num_gt_ids = set()
        for ids in raw_data['gt_ids']:
            num_gt_ids.update(ids)

        data = {
            'num_timesteps': raw_data['num_timesteps'],

            'gt_ids': raw_data['gt_ids'],
            'gt_dets': raw_data['gt_dets'],
            'gt_classes': raw_data['gt_classes'],
            'gt_crowd_ignore_regions': raw_data['gt_crowd_ignore_regions'],
            'gt_extras': raw_data['gt_extras'],

            'tracker_ids': raw_data['tracker_ids'],
            'tracker_confidences': raw_data['tracker_confidences'],
            'tracker_dets': raw_data['tracker_dets'],
            'tracker_classes': raw_data['tracker_classes'],

            'num_tracker_dets': sum(len(x) for x in raw_data['tracker_ids']),
            'num_gt_dets': sum(len(x) for x in raw_data['gt_ids']),
            'num_tracker_ids': max(num_tracker_ids) + 1,
            'num_gt_ids': max(num_gt_ids) + 1,

            'similarity_scores': raw_data['similarity_scores'],
        }

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores
