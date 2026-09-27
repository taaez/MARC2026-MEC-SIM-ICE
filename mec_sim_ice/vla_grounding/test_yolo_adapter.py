"""Pure unit tests for the command-scoped low-confidence YOLO probe."""

from __future__ import annotations

import numpy as np

from command_parser import parse_command
from grounding import Detection
from yolo_adapter import infer_cameras, on_target_probe_augment, predict_result


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


class _PlainYolo:
    """ultralytics 처럼 conf 미만 박스를 내부에서 먼저 버린다 (안 넘기면 기본 0.25)."""
    names = {3: "umbrella", 11: "bench"}

    def __init__(self, raw):
        self.raw = raw
        self.conf_seen = None

    def __call__(self, image, imgsz, verbose, conf=0.25):
        self.conf_seen = conf
        res = _Result([b for b in self.raw if float(b.conf[0]) >= conf])
        res.names = self.names
        return [res]


class _Routed:
    """RoutedDetector 흉내. id 는 DET_CLASSES 인덱스 (umbrella=3, bench=11, sunblock=2)."""
    small_to_det = np.array([2, 8, 7])
    owned = {2, 7, 8}

    def __init__(self, rows):
        self.rows = rows

    def predict(self, image):
        return (np.array([r[2] for r in self.rows], dtype=float),
                np.array([r[1] for r in self.rows], dtype=float),
                np.array([r[0] for r in self.rows], dtype=int))


def check_class_thresholds(image) -> None:
    # B 단독: YOLO 내부 컷이 클래스별 문턱(0.20)까지 내려가야 umbrella 0.22 가 살아남는다.
    plain = _PlainYolo([_Box(3, 0.22, (10, 10, 20, 20)), _Box(3, 0.15, (30, 10, 40, 20)),
                        _Box(11, 0.22, (50, 10, 90, 40)), _Box(11, 0.30, (100, 10, 140, 40))])
    got = infer_cameras(plain, {"cam": image}, min_conf=0.25)["cam"]
    assert plain.conf_seen == 0.20
    assert sorted((d.label, d.score) for d in got) == [("bench", 0.30), ("umbrella", 0.22)]

    # 라우팅: B 담당 클래스는 같은 문턱, 소형 담당(sunblock)은 B 문턱을 다시 걸지 않는다.
    routed = _Routed([(3, 0.22, (10, 10, 20, 20)), (3, 0.15, (30, 10, 40, 20)),
                      (11, 0.22, (50, 10, 90, 40)), (11, 0.30, (100, 10, 140, 40)),
                      (2, 0.21, (200, 10, 210, 20))])
    got = infer_cameras(routed, {"cam": image}, min_conf=0.25)["cam"]
    assert sorted((d.label, d.score) for d in got) == [
        ("bench", 0.30), ("sunblock", 0.21), ("umbrella", 0.22)]

    # 기록용 경로: 라우팅이어도 B 담당 클래스는 conf 로 자르고 소형 담당은 둔다.
    res = predict_result(routed, image, conf=0.35)
    assert sorted(res.names[int(b.cls[0])] for b in res.boxes) == ["sunblock"]


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

    check_class_thresholds(image)

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
