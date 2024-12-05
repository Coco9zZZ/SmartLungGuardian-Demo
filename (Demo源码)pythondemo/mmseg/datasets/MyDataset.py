# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('back', 'lung'),
        palette=[[0,0,0],[255,255,255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
#
# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset
# import os.path as osp
#
# # 将 MyDataset 类注册到 DATASETS 里
# @DATASETS.register_module()
# class MyDataset(BaseSegDataset):
#     METAINFO = dict(
#     # 数据集标注的各类名称，即 0, 1, 2, 3... 各个类别的对应名称
#     classes = ('background','lung'),
#     # 各类类别的 BGR 三通道值，用于可视化预测结果
#     PALETTE = [[0,0,0],[1,0,0]]
#     )
#     # 图片和对应的标注，这里对应的文件夹下均为 .png 后缀
#     # def __init__(self, **kwargs):
#     #     super(MyDataset, self).__init__(
#     #         img_suffix='.png',
#     #         seg_map_suffix='.png',
#     #         reduce_zero_label=False,  # 此时 label 里的 0（上面 CLASSES 里第一个 “label_a”）在计算损失函数和指标时不会被忽略。
#     #         **kwargs)
#     def __init__(self,
#                  img_suffix='.png',
#                  classes=('background', 'lung'),
#                  seg_map_suffix='.png',
#                  reduce_zero_label=False,
#                  **kwargs) -> None:
#         super().__init__(
#             classes=classes,
#             img_suffix=img_suffix,
#             seg_map_suffix=seg_map_suffix,
#             reduce_zero_label=reduce_zero_label,
#             **kwargs)