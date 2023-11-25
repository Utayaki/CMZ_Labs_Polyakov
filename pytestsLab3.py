from unittest import TestCase
import numpy as np
import torch

import numpy as np
import torch


class BloatWareConvTranspose2D:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        dtype=None,
    ):
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

        if isinstance(output_padding, int) and output_padding > -1:
            self.output_padding = output_padding
        else:
            raise ValueError("Invalid output_padding")

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

        if (
            self.output_padding >= self.dilation and self.output_padding >= self.stride
        ) or (
            self.output_padding >= self.dilation and self.output_padding >= self.stride
        ):
            raise ValueError("output_padding should be smaller than dilation or stride")

        if bias == True:
            self.bias = torch.rand(self.out_channels)
        else:
            self.bias = torch.zeros(self.out_channels)

        if isinstance(kernel_size, int):
            self.weight = torch.rand(
                self.in_channels,
                self.out_channels,
                kernel_size,
                kernel_size,
            )
            self.kernel_size = kernel_size
        else:
            raise ValueError("kernel size must be int or tuple")

        self.dtype = dtype

    def forward(self, input_tensor):
        result = []

        for l in range(self.out_channels):
            feature_map = torch.zeros(
                (input_tensor.shape[1] - 1) * self.stride
                + self.dilation * (self.kernel_size - 1)
                + 1,
                (input_tensor.shape[2] - 1) * self.stride
                + self.dilation * (self.kernel_size - 1)
                + 1,
            )  # генерация пустой feature-map
            for c in range(self.in_channels):
                # проход по всем пикселям изображения
                for i in range(0, input_tensor.shape[1]):
                    for j in range(0, input_tensor.shape[2]):
                        val = input_tensor[c][i][j]
                        proizv = val * self.weight[c][l]

                        zero_tensor = torch.zeros(
                            (self.weight.shape[2] - 1) * self.dilation + 1,
                            (self.weight.shape[3] - 1) * self.dilation + 1,
                        )

                        for a in range(0, zero_tensor.shape[0], self.dilation):
                            for b in range(0, zero_tensor.shape[1], self.dilation):
                                zero_tensor[a][b] = proizv[a // self.dilation][
                                    b // self.dilation
                                ]

                        res = np.add(
                            (zero_tensor),
                            feature_map[
                                i * self.stride : i * self.stride
                                + (self.weight.shape[2] - 1) * self.dilation
                                + 1,
                                j * self.stride : j * self.stride
                                + (self.weight.shape[3] - 1) * self.dilation
                                + 1,
                            ],
                        )
                        feature_map[
                            i * self.stride : i * self.stride
                            + (self.weight.shape[2] - 1) * self.dilation
                            + 1,
                            j * self.stride : j * self.stride
                            + (self.weight.shape[3] - 1) * self.dilation
                            + 1,
                        ] = res

            result.append(
                np.add(feature_map, np.full((feature_map.shape), self.bias[l]))
            )

        for t in range(len(result)):
            if self.output_padding > 0:
                pad_func = torch.nn.ConstantPad1d(
                    (0, self.output_padding, 0, self.output_padding), 0
                )
                result[t] = pad_func(result[t])

            result[t] = result[t][
                0 + self.padding : result[t].shape[0] - self.padding,
                0 + self.padding : result[t].shape[1] - self.padding,
            ]

        return np.array(result)


class TryTesting(TestCase):
    def test1(self):
        torchConvTranspose2D = torch.nn.ConvTranspose2d(
            15, 5, 3, stride=1, padding=1, dilation=1
        )
        input_image = torch.randn(15, 10, 10)
        output = torchConvTranspose2D(input_image)

        myConvTranspose2D = BloatWareConvTranspose2D(
            15, 5, 3, stride=1, padding=1, dilation=1
        )

        myConvTranspose2D.weight = torchConvTranspose2D.weight.detach().numpy()
        myConvTranspose2D.bias = torchConvTranspose2D.bias.detach().numpy()

        output_mock = myConvTranspose2D.forward(input_image)

        result_test = output.detach().numpy().astype("float16") == output_mock.astype(
            "float16"
        )

        self.assertTrue(any(result_test.flatten().tolist()))

    def test2(self):
        torchConvTranspose2D = torch.nn.ConvTranspose2d(
            3, 5, 1, stride=1, padding=2, dilation=1
        )
        input_image = torch.randn(3, 10, 10)
        output = torchConvTranspose2D(input_image)

        myConvTranspose2D = BloatWareConvTranspose2D(
            3, 5, 1, stride=1, padding=2, dilation=1
        )

        myConvTranspose2D.weight = torchConvTranspose2D.weight.detach().numpy()
        myConvTranspose2D.bias = torchConvTranspose2D.bias.detach().numpy()

        output_mock = myConvTranspose2D.forward(input_image)

        result_test = output.detach().numpy().astype("float16") == output_mock.astype(
            "float16"
        )

        self.assertTrue(any(result_test.flatten().tolist()))

    def test3(self):
        torchConvTranspose2D = torch.nn.ConvTranspose2d(
            32, 8, 1, stride=3, padding=2, dilation=2
        )
        input_image = torch.randn(32, 20, 20)
        output = torchConvTranspose2D(input_image)

        myConvTranspose2D = BloatWareConvTranspose2D(
            32, 8, 1, stride=3, padding=2, dilation=2
        )

        myConvTranspose2D.weight = torchConvTranspose2D.weight.detach().numpy()
        myConvTranspose2D.bias = torchConvTranspose2D.bias.detach().numpy()

        output_mock = myConvTranspose2D.forward(input_image)

        result_test = output.detach().numpy().astype("float16") == output_mock.astype(
            "float16"
        )

        self.assertTrue(any(result_test.flatten().tolist()))


def BloatWareCursedTranspConv2d(
    matrix,
    in_channels,
    out_channels,
    kernel_size,
    transp_stride=1,
    padding=0,
    dilation=1,
    bias=True,
    padding_mode="zeros",
):
    stride = 1

    # добавление отступов и padding в входной матрице
    pad = kernel_size - 1
    result_matrix = []
    for matr in matrix:
        zero_tensor = np.zeros(
            (
                ((matr.shape[0] - 1) * (transp_stride) + 1),
                ((matr.shape[1] - 1) * (transp_stride) + 1),
            )
        )
        for a in range(0, zero_tensor.shape[0], transp_stride):
            for b in range(0, zero_tensor.shape[1], transp_stride):
                zero_tensor[a][b] = matr[a // (transp_stride)][b // (transp_stride)]

        pad_matr = np.pad(zero_tensor, pad_width=pad, mode="constant")
        result_matrix.append(pad_matr)
    matrix = torch.tensor(result_matrix)

    # генерация bias
    if bias == True:
        bias_val = torch.rand(out_channels)
    else:
        bias_val = torch.zeros(out_channels)

    # padding_mode
    if padding_mode == "zeros":
        pad = torch.nn.ZeroPad2d(padding)
        matrix = pad(matrix)
    if padding_mode == "reflect":
        pad = torch.nn.ReflectionPad2d(padding)
        matrix = pad(matrix)
    if padding_mode == "replicate":
        pad = torch.nn.ReplicationPad2d(padding)
        matrix = pad(matrix)
    if padding_mode == "circular":
        pad = torch.nn.CircularPad2d(padding)
        matrix = pad(matrix)

    # генерация ядра
    filter = np.array(torch.rand(out_channels, in_channels, kernel_size, kernel_size))

    # инвертирование ядра для ConvTranspose2d
    filter_for_transpose = []
    for j in range(out_channels):
        filter_in = []
        for i in range(in_channels):
            filter_in.append(np.flip(np.array(filter[j][i])))
        filter_for_transpose.append(filter_in)

    filter_for_transpose = torch.tensor(filter_for_transpose)
    filter_for_transpose = filter_for_transpose.reshape(
        in_channels, out_channels, kernel_size, kernel_size
    )

    result = []
    for l in range(out_channels):
        feature_map = np.array([])  # генерация пустой feature-map
        for i in range(
            0, matrix.shape[1] - ((filter.shape[2] - 1) * dilation + 1) + 1, stride
        ):  # (filter.size - 1)*dilation + 1 при delation
            for j in range(
                0, matrix.shape[2] - ((filter.shape[3] - 1) * dilation + 1) + 1, stride
            ):
                summa = 0
                for c in range(in_channels):
                    val = matrix[c][
                        i : i + (filter.shape[2] - 1) * dilation + 1 : dilation,
                        j : j + (filter.shape[3] - 1) * dilation + 1 : dilation,
                    ]
                    mini_sum = (val * filter[l][c]).sum()
                    summa = summa + mini_sum
                feature_map = np.append(feature_map, float(summa + bias_val[l]))  # bias
        result.append(
            feature_map.reshape(
                (matrix.shape[1] - ((filter.shape[2] - 1) * dilation + 1)) // stride
                + 1,
                (matrix.shape[2] - ((filter.shape[3] - 1) * dilation + 1)) // stride
                + 1,
            )
        )

    return (
        np.array(result),
        torch.tensor(np.array(filter_for_transpose)),
        torch.tensor(np.array(bias_val)),
    )
