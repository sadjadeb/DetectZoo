"""DARTS genotype helpers for Raw-PC-DARTS."""

from __future__ import annotations

from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

# Paper best architecture (mel-scale sinc), from upstream README.
DEFAULT_GENOTYPE = Genotype(
    normal=[
        ("dil_conv_5", 1),
        ("dil_conv_3", 0),
        ("dil_conv_5", 1),
        ("dil_conv_5", 2),
        ("std_conv_5", 2),
        ("skip_connect", 3),
        ("std_conv_5", 2),
        ("skip_connect", 4),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("max_pool_3", 0),
        ("std_conv_3", 1),
        ("dil_conv_3", 0),
        ("dil_conv_3", 2),
        ("skip_connect", 0),
        ("dil_conv_5", 2),
        ("dil_conv_3", 0),
        ("avg_pool_3", 1),
    ],
    reduce_concat=range(2, 6),
)

DEFAULT_GENOTYPE_STRING = (
    "Genotype(normal=[('dil_conv_5', 1), ('dil_conv_3', 0), ('dil_conv_5', 1), "
    "('dil_conv_5', 2), ('std_conv_5', 2), ('skip_connect', 3), ('std_conv_5', 2), "
    "('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3', 0), "
    "('std_conv_3', 1), ('dil_conv_3', 0), ('dil_conv_3', 2), ('skip_connect', 0), "
    "('dil_conv_5', 2), ('dil_conv_3', 0), ('avg_pool_3', 1)], reduce_concat=range(2, 6))"
)
