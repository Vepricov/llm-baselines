"""TwoPhaseDataReader: transparent wrapper around two DataReaders.

Drop-in for the existing DataReader. Switches from reader1 (D_1) to reader2 (D_2)
once the internal step counter passes switch_step.

Append this to src/data/utils.py.
"""

import os


class TwoPhaseDataReader:
    def __init__(self, reader1, reader2, switch_step):
        self.reader1 = reader1
        self.reader2 = reader2
        self.switch_step = switch_step
        self._cur_step = 0

    def sample_batch(self):
        if self._cur_step < self.switch_step:
            xy = self.reader1.sample_batch()
        else:
            xy = self.reader2.sample_batch()
        self._cur_step += 1
        return xy

    def set_step(self, step):
        self._cur_step = step
        if step < self.switch_step:
            self.reader1.set_step(step)
            self.reader2.set_step(0)
        else:
            self.reader1.set_step(self.switch_step)
            self.reader2.set_step(step - self.switch_step)

    @property
    def step(self):
        return self._cur_step

    @step.setter
    def step(self, value):
        self.set_step(value)

    @property
    def num_tokens(self):
        return self.reader1.num_tokens + self.reader2.num_tokens

    def __len__(self):
        return len(self.reader1) + len(self.reader2)

    def num_batches(self):
        return self.reader1.num_batches() + self.reader2.num_batches()


def get_fineweb_twophase_paths(datasets_dir):
    """Phase 1 = fineweb-100BT, Phase 2 = fineweb-edu-100BT-subset5B.

    Both already tokenized as train.bin/val.bin. Val taken from phase 2 (D_2),
    since target is F_2 (downstream / curated risk).
    """
    fw_dir = os.path.join(datasets_dir, "fineweb-100BT")
    edu_dir = os.path.join(datasets_dir, "fineweb-edu-100BT-subset5B")
    return {
        "train_phase1": os.path.join(fw_dir, "train.bin"),
        "train_phase2": os.path.join(edu_dir, "train.bin"),
        "val": os.path.join(edu_dir, "val.bin"),
    }
