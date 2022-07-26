# Copyright (c) OpenMMLab. All rights reserved.
from .ce_loss import CELoss, SARLoss, TFLoss
from .ctc_loss import CTCLoss
from .mix_loss import ABILoss
from .seg_loss import SegLoss
from .cc_loss import CCLoss

__all__ = ['CELoss', 'SARLoss', 'CTCLoss', 'TFLoss', 'SegLoss', 'ABILoss', 'CCLoss']
