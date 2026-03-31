# ═══════════════════════════════════════════════════════════
# IK 계산 (기존 코드 유지)
# ═══════════════════════════════════════════════════════════
print("\n역기구학(IK) 계산 중...")
T_targets = {k: pose_to_T(v) for k,v in poses.items()}
q_prev = np.deg2rad([90,-110,100,-80,-90,-90])
q_solutions = {}

for name in labels:
    q_sol = ik_solve(T_targets[name], q0=q_prev)
    q_solutions[name] = q_sol
    q_prev = q_sol.copy()
    print(f"  {name}: {np.round(np.rad2deg(q_sol),1)}")

# ═══════════════════════════════════════════════════════════
# 보간 경로 생성 (기존 코드 유지)
# ═══════════════════════════════════════════════════════════
def interpolate_joint_path(q_list, n_per_segment=20):
    frames = []
    for i in range(len(q_list)-1):
        q0=q_list[i]; q1=q_list[i+1]
        dq=wrap_to_pi(q1-q0)
        for t in np.linspace(0,1,n_per_segment,endpoint=False):
            frames.append(wrap_to_pi(q0+t*dq))
    frames.append(q_list[-1])
    return frames

q_list = [q_solutions[name] for name in labels]
frames_q = interpolate_joint_path(q_list, n_per_segment=20)
print(f"\n총 {len(frames_q)}개 보간 프레임 생성 완료")

# ═══════════════════════════════════════════════════════════
# ★★★ RoboDK API — 타겟 생성 및 시뮬레이션 (변환된 부분) ★★★
# ═══════════════════════════════════════════════════════════

# ── 기준 좌표계 가져오기 ────────────────────────────────────
ref_frame = robot.Parent()

# ── 기존 타겟 초기화 ────────────────────────────────────────
print("\nRoboDK에 타겟 생성 중...")
robodk_targets = {}

for name in labels:
    q_deg = np.rad2deg(q_solutions[name]).tolist()

    # RoboDK 타겟 생성
    target = RDK.AddTarget(name, ref_frame, robot)
    target.setAsJointTarget()               # 관절 타겟으로 설정
    target.setJoints(q_deg)                 # 관절 각도 설정 (도 단위)

    robodk_targets[name] = target
    print(f"  타겟 생성: {name} → {[round(v,1) for v in q_deg]}")

# ── RoboDK 프로그램 생성 ────────────────────────────────────
print("\nRoboDK 프로그램 생성 중...")
prog = RDK.AddProgram('UR5e_IK_A_to_H', robot)

prog.setSpeed(200)          # TCP 속도 (mm/s)
prog.setSpeedJoints(30)     # 관절 속도 (deg/s)
prog.setAcceleration(500)   # 가속도

# 각 타겟에 MoveJ 명령 추가
for name in labels:
    prog.MoveJ(robodk_targets[name])
    print(f"  MoveJ 추가: {name}")

print("\n프로그램 생성 완료!")

# ═══════════════════════════════════════════════════════════
# ★ 시뮬레이션 직접 실행 (관절 각도 순차 전송) ★
# ═══════════════════════════════════════════════════════════
print("\n" + "="*45)
print("  RoboDK 시뮬레이션 시작!")
print("  로봇이 A → H 순서로 이동합니다")
print("="*45)

# 렌더링 활성화
RDK.Render(True)

# 보간된 프레임을 RoboDK에 순차 전송
DELAY = 0.03    # ← 프레임 간격 (초), 줄이면 빨라짐

for i, q in enumerate(frames_q):
    q_deg = np.rad2deg(q).tolist()
    robot.setJoints(q_deg)          # 관절 각도 직접 설정
    RDK.Render()                    # 화면 갱신
    time.sleep(DELAY)

    # 진행 상황 출력
    if i % 20 == 0:
        pct = int(100*i/len(frames_q))
        print(f"  진행: {pct}%  ({i}/{len(frames_q)} 프레임)")

print("\n✅ 시뮬레이션 완료!")
print("   스테이션 트리에서 'UR5e_IK_A_to_H' 프로그램을 확인하세요.")
print("   F5 키로 프로그램을 다시 실행할 수 있습니다.")

# ═══════════════════════════════════════════════════════════
# FK 오차 확인 (기존 코드 유지)
# ═══════════════════════════════════════════════════════════
print("\n=== FK 위치 오차 확인 [mm] ===")
for name in labels:
    T_fk,_ = fk_ur5e(q_solutions[name])
    err = np.linalg.norm(T_fk[:3,3] - T_targets[name][:3,3])
    print(f"  {name:10s}: {err:.4f} mm")
