from unittest import TestCase
import numpy as np
import torch


class BloatWareConv2D:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype=None,
    ):
        padding_modes = ["zeros", "reflect", "circular"]
        if padding_mode not in padding_modes:
            raise ValueError("Invalid padding_mode")
        self.padding_mode = padding_mode

        if isinstance(in_channels, int) and in_channels > 0:
            self.in_channels = in_channels
        else:
            raise ValueError("Invalid in_channels")

        if isinstance(out_channels, int) and out_channels > 0:
            self.out_channels = out_channels
        else:
            raise ValueError("Invalid out_channels")

        if isinstance(groups, int) and groups > 0:
            self.groups = groups
        else:
            raise ValueError("Invalid groups")

        if isinstance(stride, int) and stride > 0:
            self.stride = stride
        else:
            raise ValueError("Invalid stride")

        if isinstance(padding, int) and padding > -1:
            self.padding = padding
        else:
            raise ValueError("Invalid padding")

        if isinstance(dilation, int) and dilation > 0:
            self.dilation = dilation
        else:
            raise ValueError("Invalid dilation")
        if not (
            (self.in_channels % self.groups == 0)
            and (self.out_channels % self.groups == 0)
        ):
            raise ValueError(
                "in_channels and out_channels must both be divisible by groups"
            )

        if bias == True:
            self.bias = torch.rand(out_channels)
        else:
            self.bias = torch.zeros(out_channels)

        if isinstance(kernel_size, tuple):
            self.weight = torch.rand(
                self.out_channels,
                self.in_channels // self.groups,
                kernel_size[0],
                kernel_size[1],
            )
        elif isinstance(kernel_size, int):
            self.weight = torch.rand(
                self.out_channels,
                self.in_channels // self.groups,
                kernel_size,
                kernel_size,
            )
        else:
            raise ValueError("kernel size must be int or tuple")

        self.dtype = dtype

    def forward(self, input_tensor):
        if self.padding_mode == "zeros":
            pad = torch.nn.ZeroPad2d(self.padding)
            input_tensor = pad(input_tensor)
        if self.padding_mode == "reflect":
            pad = torch.nn.ReflectionPad2d(self.padding)
            input_tensor = pad(input_tensor)
        if self.padding_mode == "circular":
            pad = torch.nn.CircularPad2d(self.padding)
            input_tensor = pad(input_tensor)

        result = []
        for l in range(self.out_channels):
            feature_map = np.array([])  # генерация пустой feature-map
            for i in range(
                0,
                input_tensor.shape[1]
                - ((self.weight.shape[2] - 1) * self.dilation + 1)
                + 1,
                self.stride,
            ):  # (filter.size - 1)*dilation + 1 при delation
                for j in range(
                    0,
                    input_tensor.shape[2]
                    - ((self.weight.shape[3] - 1) * self.dilation + 1)
                    + 1,
                    self.stride,
                ):
                    all_channels_sum = 0
                    for c in range(self.in_channels // self.groups):  # groups
                        if self.groups > 1:
                            val = input_tensor[
                                l * (self.in_channels // self.groups) + c
                            ][
                                i: i
                                + (self.weight.shape[2] - 1) * self.dilation
                                + 1: self.dilation,
                                j: j
                                + (self.weight.shape[3] - 1) * self.dilation
                                + 1: self.dilation,
                            ]
                        else:
                            val = input_tensor[c][
                                i: i
                                + (self.weight.shape[2] - 1) * self.dilation
                                + 1: self.dilation,
                                j: j
                                + (self.weight.shape[3] - 1) * self.dilation
                                + 1: self.dilation,
                            ]
                        channel_sum = (val * self.weight[l][c]).sum()
                        all_channels_sum += +channel_sum
                    feature_map = np.append(
                        feature_map, float(all_channels_sum + self.bias[l])
                    )  # bias

            result.append(
                feature_map.reshape(
                    (
                        input_tensor.shape[1]
                        - ((self.weight.shape[2] - 1) * self.dilation + 1)
                    )
                    // self.stride
                    + 1,
                    (
                        input_tensor.shape[2]
                        - ((self.weight.shape[3] - 1) * self.dilation + 1)
                    )
                    // self.stride
                    + 1,
                )
            )

        return np.array(result)


class TryTesting(TestCase):
    def test1(self):
        kernel_size = (2, 4)
        padding = 1
        padding_mode = "zeros"
        dilation = 1
        stride = 1
        in_channels = 4
        out_channels = 16
        bias = True
        groups = 1

        torchConv2D = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        input_image = torch.randn(in_channels, 50, 50)
        output = torchConv2D(input_image)

        myConv2D = BloatWareConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        myConv2D.weight = torchConv2D.weight.detach().numpy()
        myConv2D.bias = torchConv2D.bias.detach().numpy()

        output_mock = myConv2D.forward(input_image)

        result_test = output.detach().numpy().astype("float16") == output_mock.astype(
            "float16"
        )
        self.assertTrue(any(result_test.flatten().tolist()))

    def test2(self):
        kernel_size = (3, 3)
        padding = 0
        padding_mode = "zeros"
        dilation = 1
        stride = 1
        in_channels = 2
        out_channels = 2
        bias = False
        groups = 2

        torchConv2D = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        input_image = torch.randn(in_channels, 50, 50)
        output = torchConv2D(input_image)

        myConv2D = BloatWareConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        myConv2D.weight = torchConv2D.weight.detach().numpy()
        # myConv2D.bias = torchConv2D.bias.detach().numpy()

        output_mock = myConv2D.forward(input_image)

        result_test = output.detach().numpy().astype("float16") == output_mock.astype(
            "float16"
        )
        self.assertTrue(any(result_test.flatten().tolist()))

    def test3(self):
        kernel_size = (5, 5)
        padding = 0
        padding_mode = "reflect"
        dilation = 2
        stride = 1
        in_channels = 16
        out_channels = 16
        bias = True
        groups = 1

        torchConv2D = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        input_image = torch.randn(in_channels, 50, 50)
        output = torchConv2D(input_image)

        myConv2D = BloatWareConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        myConv2D.weight = torchConv2D.weight.detach().numpy()
        myConv2D.bias = torchConv2D.bias.detach().numpy()

        output_mock = myConv2D.forward(input_image)

        result_test = output.detach().numpy().astype("float16") == output_mock.astype(
            "float16"
        )
        self.assertTrue(any(result_test.flatten().tolist()))

    def test4(self):
        kernel_size = (1, 5)
        padding = 2
        padding_mode = "circular"
        dilation = 1
        stride = 2
        in_channels = 3
        out_channels = 8
        bias = True
        groups = 1

        torchConv2D = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        input_image = torch.randn(in_channels, 50, 50)
        output = torchConv2D(input_image)

        myConv2D = BloatWareConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        myConv2D.weight = torchConv2D.weight.detach().numpy()
        myConv2D.bias = torchConv2D.bias.detach().numpy()

        output_mock = myConv2D.forward(input_image)

        result_test = output.detach().numpy().astype("float16") == output_mock.astype(
            "float16"
        )
        self.assertTrue(any(result_test.flatten().tolist()))
