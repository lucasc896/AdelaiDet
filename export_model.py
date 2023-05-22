#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn
from adet.checkpoint import AdetCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import build_model

# from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from train_damage import setup, Trainer


# experimental. API not yet final
def export_scripting(torch_model):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    class ScriptableAdapter(ScriptableAdapterBase):
        def forward(
            self, inputs: Dict[str, torch.Tensor]
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            _instances = self.model.forward([inputs])
            _instance = _instances[0]["instances"].get_fields()
            return (
                _instance["pred_boxes"].to("cpu"),
                _instance["scores"].to("cpu"),
                _instance["pred_classes"].to("cpu"),
                _instance["pred_masks"].to("cpu"),
            )

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, args.output)
    # TODO inference in Python now missing postprocessing glue code
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup(args)

    # create a torch model
    torch_model = Trainer.build_model(cfg)
    # DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    AdetCheckpointer(torch_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    torch_model.eval()

    # convert and save model
    exported_model = export_scripting(torch_model)
    logger.info("Success.")
