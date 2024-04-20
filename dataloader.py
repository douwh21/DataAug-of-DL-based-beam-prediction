from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io as sio
import torch

DATACOUNT_PER_FILE = 256

class Dataloader:
    # initialization
    def __init__(self, path="", batch_size=160, device="cpu", datacount=40960):
        self.batch_size = int(batch_size / 10)
        self.device = device
        self.filenum = int(datacount / 10) // DATACOUNT_PER_FILE 
        self.datanum = int(datacount / 10)  % DATACOUNT_PER_FILE
        # count file names
        self.files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        for i, f in enumerate(self.files):
            if not f.split(".")[-1] == "mat":
                del self.files[i]

        #for training
        if "train" in path:
            self.files.sort()
            if self.datanum == 0:
                self.files = self.files[0:self.filenum]
            else:
                self.files = self.files[0:self.filenum + 1]
                
        self.reset()

    # reset buffers
    def reset(self):
        self.done = False
        self.unvisited_files = [f for f in self.files]
        self.buffer = np.zeros((0, 2, 10, 16))
        self.buffer_label_nonoise_m = np.zeros((0, 10))
        self.buffer_beam_power_nonoise_m = np.zeros((0, 10, 64))

    # load all the data from one file
    def load(self, file):
        data = sio.loadmat(file)

        # channel: proposed by using wide beams
        channel = data["channel_data2"]
        channel = np.transpose(channel, (1, 0, 2, 3))
        # evaluation label with no noise
        labels_nonoise = data["max_id_sery_no_noise_m"] - 1
        # evaluation beam power with no noise
        beam_power_nonoise_m = data["rsrp_sery_no_noise_m"]

        # for training
        if self.datanum!=0 and len(self.unvisited_files) == 0:
            channel = channel[0 : 0 + self.datanum]
            labels_nonoise = labels_nonoise[0 : 0 + self.datanum]
            beam_power_nonoise_m = beam_power_nonoise_m[0 : 0 + self.datanum]

        return (
            channel,
            beam_power_nonoise_m,
            labels_nonoise,
        )

    def next_batch(self):
        # serial load data
        done = False
        count = True
        while self.buffer.shape[0] < self.batch_size:
            # if finishing load data
            if len(self.unvisited_files) == 0:
                done = True
                count = False
                break
            (
                channel,
                beam_power_nonoise_m,
                labels_nonoise,
            ) = self.load(self.unvisited_files.pop(0))

            del self.buffer
            del self.buffer_beam_power_nonoise_m
            del self.buffer_label_nonoise_m

            # define buffers
            self.buffer = np.zeros((0, 2, 10, 16))
            self.buffer_beam_power_nonoise_m = np.zeros((0, 10, 64))
            self.buffer_label_nonoise_m = np.zeros((0, 10))

            # load data into buffers
            self.buffer = np.concatenate((self.buffer, channel), axis=0)
            self.buffer_beam_power_nonoise_m = np.concatenate(
                (self.buffer_beam_power_nonoise_m, beam_power_nonoise_m), axis=0
            )
            self.buffer_label_nonoise_m = np.concatenate(
                (self.buffer_label_nonoise_m, labels_nonoise), axis=0
            )

        # get data from buffers
        out_size = min(self.batch_size, self.buffer.shape[0])
        batch_channel = self.buffer[0:out_size, :, :, :]
        batch_beam_power_nonoise_m = np.squeeze(
            self.buffer_beam_power_nonoise_m[0:out_size, :, :]
        )
        batch_labels_nonoise_m = np.squeeze(self.buffer_label_nonoise_m[0:out_size, :])

        self.buffer = np.delete(self.buffer, np.s_[0:out_size], 0)
        self.buffer_beam_power_nonoise_m = np.delete(
            self.buffer_beam_power_nonoise_m, np.s_[0:out_size], 0
        )
        self.buffer_label_nonoise_m = np.delete(
            self.buffer_label_nonoise_m, np.s_[0:out_size], 0
        )

        # format transformation for reducing overhead
        batch_channel = np.float32(batch_channel)
        batch_beam_power_nonoise_m = np.float32(batch_beam_power_nonoise_m)
        batch_labels_nonoise_m = batch_labels_nonoise_m.astype(np.int64)

        # return data
        return (
            torch.from_numpy(batch_channel).to(self.device),
            torch.from_numpy(batch_beam_power_nonoise_m).to(self.device),
            torch.from_numpy(batch_labels_nonoise_m).to(self.device),
            done,
            count,
        )
