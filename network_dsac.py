import torch
import torch.nn as nn


class DsacNet(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean=torch.zeros((3,)), tiny=0):
        """
        Constructor.
        """
        super(DsacNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, (256, 128)[tiny], 3, 2, 1)

        self.res1_conv1 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)
        self.res1_conv2 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 1, 1, 0)
        self.res1_conv3 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)

        self.res2_conv1 = nn.Conv2d((256, 128)[tiny], (512, 128)[tiny], 3, 1, 1)
        self.res2_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res2_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 3, 1, 1)

        if not tiny:
            self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)

        self.res3_conv1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)

        self.fc1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc3 = nn.Conv2d((512, 128)[tiny], 3, 1, 1, 0)

        # learned scene coordinates relative to a mean coordinate (e.g. center of the scene)
        self.register_buffer("mean", torch.tensor(mean.size()).cuda())
        self.mean = mean.clone()
        self.tiny = tiny
        # self.crf = CRF(n_spatial_dims=2)

    def forward(self, inputs):
        """
        Forward pass.

        inputs -- 4D data tensor (BxCxHxW)
        """

        x = inputs
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        res = torch.relu(self.conv4(x))

        x = torch.relu(self.res1_conv1(res))
        x = torch.relu(self.res1_conv2(x))
        x = torch.relu(self.res1_conv3(x))

        res = res + x

        x = torch.relu(self.res2_conv1(res))
        x = torch.relu(self.res2_conv2(x))
        x = torch.relu(self.res2_conv3(x))

        if not self.tiny:
            res = self.res2_skip(res)

        res = res + x

        x = torch.relu(self.res3_conv1(res))
        x = torch.relu(self.res3_conv2(x))
        x = torch.relu(self.res3_conv3(x))

        res = res + x

        sc = torch.relu(self.fc1(res))
        sc = torch.relu(self.fc2(sc))
        sc = self.fc3(sc)

        sc[:, 0] += self.mean[0]
        sc[:, 1] += self.mean[1]
        sc[:, 2] += self.mean[2]
        # sc = self.crf(sc)

        return sc

    def sample_sc_slow(self, sc, coordinates, gt=False):
        sc = sc.cpu()
        res = []
        for ind, coord in enumerate(coordinates):
            if not gt:
                coord = ((coord - 4) / 8).long()
            else:
                coord = coord.long()
            points = torch.zeros((coord.shape[0], 3))
            for ind2, (x, y) in enumerate(coord):
                points[ind2] = sc[ind, :, y, x]

            res.append(points)
        return res

    def sample_sc(self, sc, coordinates, gt=False, switch_channel=True):
        res = []
        for ind, coord in enumerate(coordinates):
            if not gt:
                coord = ((coord - 4) / 8).long()
            else:
                coord = coord.long()
            if switch_channel:
                points2 = sc[ind, :, coord[:, 1], coord[:, 0]]
            else:
                points2 = sc[ind, :, coord[:, 0], coord[:, 1]]

            res.append(points2)
        return res

