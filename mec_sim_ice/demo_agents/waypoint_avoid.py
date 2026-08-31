"""웨이포인트 원둘레 회피 -- 경로를 통째로 재계획하지 않고 침범당한 구간만 휜다.

## 왜 필요한가

`participant_app._navigate_to()` 의 기존 회피는 두 갈래뿐이었다: 숙고형 A* 재계획
(격자 갱신 + A* + `PathFollower` 교체, 최소 한 틱 지연)과 매틱 반응형 제동(`linear=0`
만, 조향은 안 건드림). 근접 장애물이 재계획 사이클을 기다리는 동안 제동이 시간을 버는
구조였는데, 제동은 "회피"가 아니라 "숙고 경로가 착지할 시간을 버는 것"뿐이라 제동
자체가 오래 걸리면 (`BRAKE_STALL_S`) 정체로 취급해야 했다.

이 모듈은 세 번째 갈래를 더한다: 장애물이 원(중심, 반지름)으로 근사된다는 점을
이용해, 경로가 원을 관통하는 구간만 원 둘레를 따라 도는 웨이포인트로 **그 자리에서**
치환한다. `PathFollower.path` 를 직접 수정하는 연산이라 재계획 사이클이 없다 -- 격자
갱신도, A* 도, follower 교체도 필요 없다.

## 원 표현과의 관계

`lidar_obstacles.LidarObstacleLayer.clusters()` 가 라이다 차단 셀을 장애물 개체별
원으로 묶어 낸다(20260818_임우석_발표자료의 DBSCAN 클러스터링과 같은 목적). 정체
기억(`participant_app._obstacles`)도 이미 원 표현이다. 이 모듈은 그 원이 어디서 왔는지
모른다 -- `[(x, y, r), ...]` 만 받는다.

## 한계 (의도적으로 안 만든 것)

- 한 세그먼트가 여러 원을 동시에 관통하면 **가장 먼저 만나는 원 하나만** 처리하고,
  삽입된 우회 구간은 나머지 원과 다시 검사하지 않는다. Stage2 의 장애물 밀도(주차
  차량·자전거 몇 개)에서 세그먼트 하나가 겹친 원 여러 개를 동시에 관통하는 경우는
  드물고, 실패하면(둘레가 양쪽 다 안 열려 있으면) 호출자가 전체 A* 재계획으로
  폴백하므로 안전은 유지된다. 실측에서 자주 걸리면 그때 재귀적으로 확장한다.
- 이미 지나간 웨이포인트(`follower.idx` 이전)는 건드리지 않는다 -- 회피는 항상 앞으로
  갈 구간에 대해서만.
"""

from __future__ import annotations

import math

ARC_STEP_DEG = 20.0   # 원 둘레를 따라 넣는 중간점의 각도 간격

# 겹친 원(길쭉한 장애물을 쪼갠 이웃 원들)을 제대로 다루려면, 방금 넣은 우회 구간도
# 건너뛰지 않고 다른 원들과 다시 검사해야 한다(아래 메인 루프). 문제는 원 둘레를 따라
# 찍은 각 세그먼트(연속 호점 사이, 또는 진입점 자신을 잇는 첫 구간)가 **자기 자신을
# 만든 그 원**과의 검사에서 가짜로 "관통"처럼 보인다는 것: 원 위의 두 점을 잇는 직선은
# 두 끝에서 거리==r 인 근을 정확히 갖는데(현), 원의 정의상 현은 항상 원 내부를 지나기
# 때문이다. 그래서 "근이 두 개 존재하는가"가 아니라 "세그먼트가 원 중심에 실제로
# 얼마나 파고드는가"로 판정한다 -- 파고든 깊이가 이 호 근사 자체의 새지타(sagitta,
# 현이 둘레보다 안쪽으로 들어간 깊이)보다 작으면 진짜 관통이 아니라 우리가 방금 그린
# 근사 자체로 보고 무시한다. 안전 배수 2 를 곱해 부동소수점 잡음도 함께 흡수한다.
ARC_SAGITTA_SAFETY = 2.0

# 원 여러 개가 겹쳐 있을 때(길쭉한 장애물을 쪼갠 이웃 원들) 우회가 우회를 유발하며
# 무한히 이어지지 않도록 하는 안전판. 정상적인 경우 삽입 횟수는 원 개수 정도에서
# 끝난다 -- 이 상한을 넘으면 기하가 병적이라는 뜻이므로 포기하고 호출자가 전체
# 재계획으로 폴백하게 한다.
MAX_INSERTIONS = 200


def _min_dist_to_segment(center, a, b):
    """점 center 에서 선분 a-b 까지의 최단 거리."""
    ax, ay = a
    bx, by = b
    cx, cy = center
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(cx - ax, cy - ay)
    t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(cx - px, cy - py)


def _circle_segment_crossings(a, b, center, r):
    """선분 a->b 가 원(center, r) 을 지나는 t in [0, 1] 값들 (0, 1, 또는 2개, 오름차순)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - center[0], a[1] - center[1]
    A = dx * dx + dy * dy
    if A < 1e-12:
        return []
    B = 2.0 * (fx * dx + fy * dy)
    C = fx * fx + fy * fy - r * r
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)
    return sorted(t for t in (t1, t2) if 0.0 <= t <= 1.0)


def _arc_span(ang_from, ang_to, ccw):
    """ang_from -> ang_to 로 도는 부호 있는 각도(라디안). ccw=False 면 반대 방향."""
    if ccw:
        return (ang_to - ang_from) % (2.0 * math.pi)
    return -((ang_from - ang_to) % (2.0 * math.pi))


def _arc_points(center, radius, ang_from, span):
    """원 둘레 위의 중간점들 (양 끝점은 제외 -- 호출자가 entry/exit 을 따로 갖고 있음)."""
    n = max(1, int(math.ceil(abs(span) / math.radians(ARC_STEP_DEG))))
    pts = []
    for i in range(1, n):
        a = ang_from + span * i / n
        pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
    return pts


def _fully_clear(points, circles, tol=1e-6):
    """모든 점이 모든 원 밖(경계 포함)에 있는가."""
    return all(math.hypot(px - cx, py - cy) >= r - tol
               for (px, py) in points for (cx, cy, r) in circles)


def _resolve_point(pt, circles, iters=3):
    """pt 를 모든 원 밖으로 밀어낸다 (겹친 원이면 몇 차례 반복해야 전부 벗어난다)."""
    for _ in range(iters):
        moved = False
        for (cx, cy, r) in circles:
            new_p = _push_out(pt, (cx, cy), r)
            if new_p != pt:
                pt = new_p
                moved = True
        if not moved:
            break
    return pt


def _push_out(pt, center, r):
    """pt 가 원 안이면 반지름 위로 방사형으로 밀어낸다. 이미 밖이면 그대로."""
    dx, dy = pt[0] - center[0], pt[1] - center[1]
    d = math.hypot(dx, dy)
    if d >= r:
        return pt
    if d < 1e-9:
        return (center[0] + r, center[1])   # 퇴화 케이스: 임의 방향
    s = r / d
    return (center[0] + dx * s, center[1] + dy * s)


def relocate(path, circles, margin_m=0.0, is_free=None):
    """경로가 원을 관통하는 구간만 원 둘레를 따라 재배치한다.

    Args:
        path: `[(x, y), ...]` 세계좌표 웨이포인트. 첫 점은 보통 로봇의 현재 위치를
            앵커로 붙인 것(호출자가 `[(x, y)] + remaining` 형태로 준다) -- 그래야
            "지금 위치에서 다음 웨이포인트로 가는 구간"의 관통도 잡힌다.
        circles: `[(x, y, r), ...]` 장애물 원 (마진 미포함, 물리적 외곽).
        margin_m: 각 원에 더할 로봇 여유(반경+틈). 실제 회피 반경은 `r + margin_m`.
        is_free: `(x, y) -> bool`, 선택. 우회 방향이 점유지도 상 자유공간인지 확인용
            -- 주어지면 짧은 호부터 시도하되 자유공간이 아니면 반대 방향을 시도한다.
            없으면(오프라인 검증 등) 항상 더 짧은 호를 택한다.

    Returns:
        재배치된 경로(list). 관통이 없으면 원본과 같은 내용의 새 리스트.
        어느 원이 양쪽 다(자유공간이 아니거나 대칭이라 시도할 다른 쪽이 없어) 못 돌면
        `None` -- 호출자는 전체 A* 재계획으로 폴백해야 한다.
    """
    if not path:
        return list(path) if path is not None else path
    if not circles:
        return list(path)

    eff = [(float(cx), float(cy), float(r) + float(margin_m)) for (cx, cy, r) in circles]
    pts = [(float(x), float(y)) for (x, y) in path]

    # 0) 이미 원 안에 들어와 있는 기존 웨이포인트를 먼저 방사형으로 밀어낸다 -- 원이
    # 스윕 사이에 자라서 예전 웨이포인트를 삼킨 경우. 원끼리 겹치면 한 번의 push 로
    # 다른 원 안에 다시 들어갈 수 있어 몇 차례 반복한다.
    pts = [_resolve_point(p, eff) for p in pts]

    # 1) 세그먼트 관통 검사 -- 관통이 발견되면 그 구간만 원 둘레 우회로 치환한다.
    i = 0
    insertions = 0
    while i < len(pts) - 1:
        a, b = pts[i], pts[i + 1]
        best = None   # (t_in, t_out, circle)
        for (cx, cy, r) in eff:
            # 새지타 필터: 원 중심까지 실제로 이 정도는 파고들어야 "진짜 관통"으로 본다
            # -- 그렇지 않으면 방금 그린 호 근사 자신과의 검사에서 항상 걸린다.
            sagitta = r * (1.0 - math.cos(math.radians(ARC_STEP_DEG) / 2.0)) * ARC_SAGITTA_SAFETY
            if r - _min_dist_to_segment((cx, cy), a, b) <= sagitta:
                continue
            ts = _circle_segment_crossings(a, b, (cx, cy), r)
            if len(ts) == 2 and (best is None or ts[0] < best[0]):
                best = (ts[0], ts[1], (cx, cy, r))
        if best is None:
            i += 1
            continue

        insertions += 1
        if insertions > MAX_INSERTIONS:
            return None   # 병적인 겹침 -- 포기하고 전체 재계획에 맡긴다

        t_in, t_out, (cx, cy, r) = best
        entry = (a[0] + (b[0] - a[0]) * t_in, a[1] + (b[1] - a[1]) * t_in)
        exitp = (a[0] + (b[0] - a[0]) * t_out, a[1] + (b[1] - a[1]) * t_out)
        ang_in = math.atan2(entry[1] - cy, entry[0] - cx)
        ang_out = math.atan2(exitp[1] - cy, exitp[0] - cx)

        candidates = []
        for ccw in (True, False):
            span = _arc_span(ang_in, ang_out, ccw)
            arc = _arc_points((cx, cy), r, ang_in, span)
            raw = [entry] + arc + [exitp]
            # 방금 그린 호가 이웃한(겹친) 다른 원 안으로 들어갈 수 있다 -- 길쭉한
            # 장애물을 쪼갠 원들은 서로 크게 겹치는 게 흔하다. 세그먼트 교차 검사는
            # 호점 하나가 다른 원 안에 있는데 양옆 이웃 호점은 밖인 경우(교차가 아니라
            # 그냥 그 점 자체가 안쪽인 경우)를 못 잡으므로, 여기서 각 점을 모든 원
            # 밖으로 직접 밀어낸다.
            resolved = [_resolve_point(p, eff) for p in raw]
            candidates.append((abs(span), resolved))
        candidates.sort(key=lambda item: item[0])   # 짧은 호부터

        detour = None
        for _span, cand in candidates:
            # 겹친 원이 심하면(3개 이상 서로 겹치는 등) _resolve_point 가 몇 번 반복해도
            # 완전히 못 벗어날 수 있다 -- 그런 후보는 자유공간 여부를 묻기 전에 먼저
            # 걸러서, 안 풀린 방향을 "우회 성공"으로 잘못 채택하지 않게 한다.
            if not _fully_clear(cand, eff):
                continue
            if is_free is None or all(is_free(px, py) for (px, py) in cand):
                detour = cand
                break
        if detour is None:
            return None   # 양쪽 다 막혔거나 못 풂 -- 호출자가 전체 재계획으로 폴백

        pts[i + 1:i + 1] = detour
        # i 는 그대로 둔다(건너뛰지 않음) -- 방금 넣은 첫 구간(a -> entry)을 다음
        # 루프 반복에서 다시 검사한다. 길쭉한 장애물을 여러 원으로 쪼갰을 때 이웃
        # 원끼리 겹치는 게 흔해서, 우회가 다른 원 안으로 착지하는 경우를 여기서
        # 잡아야 한다(sagitta 필터가 이 원 자신과의 가짜 재교차는 걸러준다).

    # 마지막 안전판 -- 위 처리로도 여전히 어떤 원 안에 점이 남아 있으면(과도하게 겹친
    # 원이 resolve 반복 한도 안에서 안 풀린 경우) 틀린 경로를 내보내는 대신 포기한다.
    if not _fully_clear(pts, eff):
        return None

    return pts
