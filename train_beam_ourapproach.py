import argparse
import os
import random
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from dataloader import Dataloader
from model_3Dcov_basic import Model_3D

parser = argparse.ArgumentParser(description="PyTorch Beam Prediction Training")
# Optimization options
parser.add_argument(
    "--epochs", default=4000, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--batch-size", default=16, type=int, metavar="N", help="train batchsize"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
# Miscs
parser.add_argument("--manualSeed", type=int, default=0, help="manual seed")
# Device options
parser.add_argument(
    "--gpu", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
# Method options
parser.add_argument("--n", type=int, default=64, help="Number of training data")
parser.add_argument("--p", default=0.5, type=float, help="the probability of input augmentation methods")
parser.add_argument("--sigma", default=0.3, type=float, help="the standard variance of noise injection")
parser.add_argument("--Zf", default=8.0, type=float, help="the maximum value of Z")
parser.add_argument("--Z0", default=2.0, type=float,help="the initial value of Z")
parser.add_argument("--k", default=6.0, type=float,help='slope coefficient')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(int(time.time()))

# Input augmentation
def inputaugmentation(data, label, beam_power, device):
    a, b, c, d = data.shape

    # Random Cyclic Shift
    if np.random.rand() < args.p:
        m = np.random.randint(0, d)
        data = data.roll(m, -1)
        label = (label + 4 * m) % 64
        beam_power = beam_power.roll(4 * m, -1)

    # Random Flip
    if np.random.rand() < args.p:
        data = torch.flip(data, dims=[-1])        
        data[:,1,:,:] *= -1
        beam_power = torch.flip(beam_power, dims=[-1])
        label = 63 - label

    # Noise injection
    if np.random.rand() < args.p:
        data += torch.from_numpy(np.random.randn(a, b, c, d) * args.sigma).to(device)

    return data, label, beam_power


# Label augmentation
def labelaugmentation(beam_power, labels,epoch, device):
    if args.Zf == 0:
        # One-hot
        newlabels = (
            torch.zeros(labels.shape[0], 10, 64)
            .to(device)
            .scatter_(2, labels.view(-1, 10, 1).long(), 1)
        )
    else:
        # Adaptive power scheduler
        Z = linear_rampup(epoch)
        newlabels = beam_power**(Z/2)
    return newlabels


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(args.Z0 + current / rampup_length * args.Zf * args.k, 0.0, args.Zf)
        return float(current)
    
class crossentropy(object):
    def __call__(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        return Lx


def eval(model, loader):
    # reset dataloader
    loader.reset()
    # loss function
    criterion = nn.CrossEntropyLoss()
    # judge whether dataset is finished
    done = False
    # counting accurate prediction
    P = 0
    # counting inaccurate prediction
    N = 0
    # beam power loss
    BL = np.zeros((10,5))
    # running loss
    running_loss = 0
    # count batch number
    batch_num = 0
    batch_size = 16

    # evaluate validation set
    while not done:
        # read files
        (
            channel2,
            beam_power_nonoise_m,
            label_nonoise_m,
            done,
            count,
        ) = loader.next_batch()
        if count==True:
            batch_num += 1
            # channel2: proposed by using wide beams
            out_tensor = model(channel2)
            loss = 0
            # average loss of all predictions
            for loss_count in range(10):
                loss += criterion(torch.squeeze(out_tensor[loss_count, :, :]), label_nonoise_m[:, loss_count])
            # predicted probabilities
            out_tensor_np = out_tensor.cpu().detach().numpy()
            # true label without noise
            gt_labels = label_nonoise_m.cpu().detach().numpy()
            gt_labels = np.float32(gt_labels)
            gt_labels = gt_labels.transpose(1, 0)
            # true normalized beam amplitude
            beam_power = beam_power_nonoise_m.cpu().detach().numpy()
            beam_power = beam_power.transpose(1, 0, 2)
            out_shape = gt_labels.shape
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    train_ans = np.squeeze(out_tensor_np[i, j, :])
                    # find the index with the maximum probability
                    train_index = np.argmax(train_ans)
                    # find the rank of the true optimal beam
                    train_sorted = np.argsort(train_ans)
                    # counting accurate and inaccurate prediction
                    if train_index == gt_labels[i, j]:
                        P = P + 1
                    else:
                        N = N + 1
                    # calculate beam power loss
                    BL[i, 0] = BL[i, 0] + (beam_power[i, j, train_index] / max(beam_power[i, j, :])) ** 2
                    BL[i, 1] = BL[i, 1] + (max(beam_power[i, j, train_sorted[62 : 64]]) / max(beam_power[i, j, :])) ** 2
                    BL[i, 2] = BL[i, 2] + (max(beam_power[i, j, train_sorted[61 : 64]]) / max(beam_power[i, j, :])) ** 2
                    BL[i, 3] = BL[i, 3] + (max(beam_power[i, j, train_sorted[60 : 64]]) / max(beam_power[i, j, :])) ** 2
                    BL[i, 4] = BL[i, 4] + (max(beam_power[i, j, train_sorted[59 : 64]]) / max(beam_power[i, j, :])) ** 2

            running_loss += loss.data.cpu()
    # average accuracy
    acur = float(P) / (P + N)
    # average loss
    losses = running_loss / batch_num
    # average loss
    BL = BL / batch_num / batch_size
    # print results
    print("Accuracy: %.3f" % (acur))
    print("Loss: %.3f" % (losses))
    print("Beam power loss:")
    print(BL.T)
    return acur, losses, BL


def main():
    # learning rate
    lr = args.lr
    version_name = (
        "supervised_{}_ourapproach_lr={}_p={}_Zf={}_Z0={}_k={}_sigma={}".format(
            args.n, args.lr,  args.p, args.Zf, args.Z0, args.k, args.sigma
        )
    )
    info = "DataAug_64beam_" + version_name
    print(info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # training time
    t = 5
    # training epoch
    epoch = args.epochs
    # batch size
    batch_size = args.batch_size
    print("batch_size:%d" % (batch_size))

    # training set and validation set
    loader = Dataloader(
        path="./dataset/training", batch_size=batch_size, device=device, n=args.n
    )
    eval_loader = Dataloader(
        path="./dataset/testing", batch_size=batch_size, device=device
    )

    # loss function
    criterion = crossentropy()

    # for evaluation
    acur_eval = np.zeros(( t, epoch))
    loss_eval = np.zeros((t, epoch))
    BL_eval = np.zeros((10, 5, t, epoch))

    # first loop for training times
    for tt in range(t):
        print("Train %d times" % (tt))

        # model initialization
        model = Model_3D(N=15, K=32, Tx=4, Channel=2)
        model.to(device)

        # save maximum beampower
        max_BL= 0
        # Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr / 10, betas=(0.9, 0.999)
        )  # use the sum of 10 losses

        # print parameters
        for name, param in model.named_parameters():
            print("Name:", name, "Size:", param.size())

        # second loop for training times
        for e in range(epoch):
            print("Train %d epoch" % (e))
            # reset the dataloader
            loader.reset()
            eval_loader.reset()
            # judge whether data loading is done
            done = False
            # running loss
            running_loss = 0
            # count batch number
            batch_num = 0
            while not done:
                # read files
                (
                    labeled_data,
                    beam_power_nonoise,
                    labels_nonoise,
                    done,
                    count,
                ) = loader.next_batch()

                if count == True:
                    batch_num += 1
                    inputs_x, targets_x, beam_power_nonoise = inputaugmentation(
                        labeled_data, labels_nonoise, beam_power_nonoise, device
                    )
                    targets_x = labelaugmentation(beam_power_nonoise, targets_x, e, device)
                    # predicted probabilities
                    out_tensor = model(inputs_x).transpose(0, 1)
                    loss = 0
                    # average loss of all predictions
                    for loss_count in range(10):
                        loss += criterion(
                            torch.squeeze(out_tensor[:, loss_count, :]),
                            targets_x[:, loss_count, :],
                        )
                    # gradient back propagation
                    loss.backward()
                    # parameter optimization
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
            losses = running_loss / batch_num
            # print results
            print("[%d] loss: %.3f" % (e + 1, losses))
            # eval mode, where dropout is off
            model.eval()
            print("the evaling set:")
            acur, losses,  BL = eval(model, eval_loader)
            acur_eval[tt, e] = np.squeeze(acur)
            loss_eval[tt, e] = losses
            BL_eval[:, :, tt, e] = np.squeeze(BL)
            #save the best model
            if np.mean(BL[:,0]) > max_BL:
                max_BL = np.mean(BL[:,0])
                model_name = info + "_" + str(tt) + "_MODEL.pkl"
                torch.save(model, model_name)


            # train mode, where dropout is on
            model.train()

    mat_name = info + "_evaluation.mat"
    sio.savemat(
        mat_name,
        {
            "acur_eval": acur_eval,
            "loss_eval": loss_eval,
            "BL_eval": BL_eval
        },
    )


if __name__ == "__main__":
    main()
