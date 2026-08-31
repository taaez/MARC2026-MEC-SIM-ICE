"""Pure unit tests for the command-scoped low-confidence YOLO probe."""

from __future__ import annotations

import numpy as np

from command_parser import parse_command
from grounding import Detection
from yolo_adapter import on_target_probe_augment


class _Box:
    def __init__(self, cls_id, score, bbox):
        self.cls = np.asarray([cls_id], dtype=float)
        self.conf = np.asarray([score], dtype=float)
        self.xyxy = np.asarray([bbox], dtype=float)


class _Result:
    names = {0: "mug", 1: "juice"}

    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def predict(self, image, **kwargs):
        self.calls += 1
        assert kwargs["conf"] == 0.001
        return [self.result]


def main() -> int:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    table = Detection("picnic_table", (100, 100, 500, 400), 0.95)
    mug_on_table = _Box(0, 0.006, (200, 120, 220, 150))
    mug_off_table = _Box(0, 0.020, (700, 500, 720, 530))
    model = _Model(_Result([mug_on_table, mug_off_table]))
    dets = {"cam": [table]}

    on_target_probe_augment(
        parse_command("Find the mug on the picnic table."),
        model, {"cam": image}, dets,
    )
    rescued = [d for d in dets["cam"] if d.label == "mug"]
    assert model.calls == 1
    assert len(rescued) == 1 and rescued[0].bbox == (200.0, 120.0, 220.0, 150.0)

    # 이미 정상 target 이 있으면 추가 forward 자체가 없어야 한다.
    model.calls = 0
    on_target_probe_augment(
        parse_command("Find the mug on the picnic table."), model,
        {"cam": image}, {"cam": [table, Detection("mug", (1, 1, 2, 2), 0.9)]},
    )
    assert model.calls == 0

    # near/beside 문항에는 낮은 probe 후보가 개입하지 않는다.
    model.calls = 0
    on_target_probe_augment(
        parse_command("Find the mug near the picnic table."), model,
        {"cam": image}, {"cam": [table]},
    )
    assert model.calls == 0

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
