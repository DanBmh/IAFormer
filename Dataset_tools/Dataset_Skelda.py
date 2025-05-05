import copy
import sys

import numpy as np
import torch.utils.data as data
import tqdm

sys.path.append("/PoseForecasters/")
import utils_pipeline

# ==================================================================================================

datamode = "gt-gt"
# datamode = "pred-pred"

config = {
    "item_step": 2,
    "window_step": 2,
    # "item_step": 1,
    # "window_step": 1,
    "select_joints": [
        "hip_right",
        "hip_left",
        "knee_right",
        "knee_left",
        "ankle_right",
        "ankle_left",
        "nose",
        "shoulder_right",
        "shoulder_left",
        "elbow_right",
        "elbow_left",
        "wrist_right",
        "wrist_left",
    ],
}

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_rpt.json",
]

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_rpt.json"


# ==================================================================================================


class Datasets(data.Dataset):
    def __init__(self, opt, mode):
        self.opt = opt
        seq_len = self.opt.seq_len

        config["input_n"] = seq_len // 3 * 2
        config["output_n"] = seq_len // 3

        # Load preprocessed datasets
        print("Loading datasets ...")
        dataset = None
        if mode == "train":
            dataset_train, dlen_train = [], 0
            for dp in datasets_train:
                cfg = copy.deepcopy(config)
                if "mocap" in dp:
                    cfg["select_joints"][
                        cfg["select_joints"].index("nose")
                    ] = "head_upper"

                ds, dlen = utils_pipeline.load_dataset(dp, "train", cfg)
                dataset_train.extend(ds["sequences"])
                dlen_train += dlen
            dataset = dataset_train
            dlen = dlen_train
        else:
            if mode != "test":
                esplit = "test" if "mocap" in dataset_eval_test else "eval"
            else:
                esplit = "test"
            cfg = copy.deepcopy(config)
            if "mocap" in dataset_eval_test:
                cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
            dataset_eval, dlen_eval = utils_pipeline.load_dataset(
                dataset_eval_test, esplit, cfg
            )
            dataset_eval = dataset_eval["sequences"]
            dataset = dataset_eval
            dlen = dlen_eval

        self.data = []
        label_gen = utils_pipeline.create_labels_generator(dataset, config)

        nbatch = 1
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen, batch_size=nbatch),
            total=int(dlen / nbatch),
        ):
            sequences_train = utils_pipeline.make_input_sequence(
                batch, "input", datamode, make_relative=False
            )
            sequences_gt = utils_pipeline.make_input_sequence(
                batch, "target", datamode, make_relative=False
            )

            # Switch y and z axes
            sequences_train = sequences_train[:, :, :, [0, 2, 1]]
            sequences_gt = sequences_gt[:, :, :, [0, 2, 1]]

            # Reshape to [nbatch, npersons, nframes, njoints, 3]
            J = len(config["select_joints"])
            sequences_train = sequences_train.reshape(
                [nbatch, 1, sequences_train.shape[1], J, 3]
            )
            sequences_gt = sequences_gt.reshape(
                [nbatch, 1, sequences_gt.shape[1], J, 3]
            )

            temp_data = np.concatenate([sequences_train, sequences_gt], axis=2)
            temp_data = temp_data[0]
            temp_data = temp_data.reshape(1, -1, J * 3)

            self.data.append(temp_data)
        self.len = len(self.data)

    def __getitem__(self, index):

        input_seq = self.data[index][:, : self.opt.frame_in, :]
        output_seq = self.data[index][:, : self.opt.seq_len, :]

        pad_idx = np.repeat([self.opt.frame_in - 1], self.opt.frame_out)
        i_idx = np.append(np.arange(0, self.opt.frame_in), pad_idx)

        input_seq = input_seq.transpose(0, 2, 1)
        input_seq = input_seq[:, :, i_idx]
        output_seq = output_seq.transpose(0, 2, 1)

        # print(input_seq.shape)
        # print(output_seq.shape)
        # # (1, 39, 75)
        # # (1, 39, 75)

        return input_seq, output_seq

    def __len__(self):
        return self.len
