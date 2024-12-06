import sympy as sp

# 時間シンボル
t = sp.symbols('t')

# 一般化座標（時間関数として定義）
q_x, q_y, q_z, q_w = [sp.Function(s)(t) for s in ['q_x', 'q_y', 'q_z', 'q_w']]
theta2 = sp.Function('theta2')(t)

# 時間微分
dq_x, dq_y, dq_z, dq_w, dtheta2 = [sp.diff(q, t) for q in [q_x, q_y, q_z, q_w, theta2]]
ddq_x, ddq_y, ddq_z, ddq_w, ddtheta2 = [sp.diff(dq, t) for dq in [dq_x, dq_y, dq_z, dq_w, dtheta2]]

# システムパラメータ
l1, l2, lg1, lg2, m1, m2, g = sp.symbols('l1 l2 lg1 lg2 m1 m2 g')
I1, I2 = sp.symbols('I1 I2')  # リンク1とリンク2の慣性モーメント

# リンク1の重心位置（クォータニオンで回転）
r1_local = sp.Matrix([lg1, 0, 0])
r1_global = sp.Matrix([
    (2*q_w**2 + 2*q_x**2 - 1) * r1_local[0] + (2*q_x*q_y - 2*q_z*q_w) * r1_local[1] + (2*q_x*q_z + 2*q_y*q_w) * r1_local[2],
    (2*q_x*q_y + 2*q_z*q_w) * r1_local[0] + (2*q_w**2 + 2*q_y**2 - 1) * r1_local[1] + (2*q_y*q_z - 2*q_x*q_w) * r1_local[2],
    (2*q_x*q_z - 2*q_y*q_w) * r1_local[0] + (2*q_y*q_z + 2*q_x*q_w) * r1_local[1] + (2*q_w**2 + 2*q_z**2 - 1) * r1_local[2]
])

# リンク1の先端位置
r1_end = sp.Matrix([
    (2*q_w**2 + 2*q_x**2 - 1) * l1,
    (2*q_x*q_y + 2*q_z*q_w) * l1,
    (2*q_x*q_z - 2*q_y*q_w) * l1
])

# リンク2の重心位置（肘の回転適用）
r2_local = sp.Matrix([lg2, 0, 0])
r2_global = r1_end + sp.Matrix([
    sp.cos(theta2) * r2_local[0] - sp.sin(theta2) * r2_local[2],
    r2_local[1],
    sp.sin(theta2) * r2_local[0] + sp.cos(theta2) * r2_local[2]
])

# リンクの速度
v1_global = sp.diff(r1_global, t)
v2_global = sp.diff(r2_global, t)

# リンク1の角速度（肩の回転を表すクォータニオン）
omega1 = 2 * sp.Matrix([
    [-q_y, q_x, q_w, -q_z],
    [-q_z, -q_w, q_x, q_y],
    [-q_w, q_z, -q_y, q_x]
]) * sp.Matrix([dq_x, dq_y, dq_z, dq_w])

# リンク2の角速度（肘回転）
omega2 = sp.Matrix([0, dtheta2, 0])

# 並進エネルギー
T = (1/2) * m1 * v1_global.dot(v1_global) + (1/2) * m2 * v2_global.dot(v2_global)

# 回転エネルギー
R = (1/2) * omega1.dot(I1 * omega1) + (1/2) * omega2.dot(I2 * omega2)

# 運動エネルギー
K = T + R

# ポテンシャルエネルギー
U = m1 * g * r1_global[2] + m2 * g * r2_global[2]

# ラグランジアン
L = K - U

# 一般化座標とその時間微分
generalized_coords = [q_x, q_y, q_z, q_w, theta2]
generalized_velocities = [dq_x, dq_y, dq_z, dq_w, dtheta2]
generalized_accelerations = [ddq_x, ddq_y, ddq_z, ddq_w, ddtheta2]

# 慣性行列、コリオリ項、重力項の抽出
M = sp.zeros(len(generalized_coords), len(generalized_coords))  # 慣性行列
C = sp.zeros(len(generalized_coords), len(generalized_coords))  # コリオリ項
G = sp.zeros(len(generalized_coords), 1)  # 重力項

# ラグランジュ方程式の構築と行列の抽出
lagrange_eqs = []
for i, (qi, qidot, qiddot) in enumerate(zip(generalized_coords, generalized_velocities, generalized_accelerations)):
    # ラグランジュ方程式を計算
    dL_dqi = sp.diff(L, qi)  # ∂L/∂qi
    dL_dqidot = sp.diff(L, qidot)  # ∂L/∂qidot
    ddt_dL_dqidot = sum(sp.diff(dL_dqidot, v) * a for v, a in zip(generalized_velocities, generalized_accelerations)) \
                    + sum(sp.diff(dL_dqidot, q) * qidot for q, qidot in zip(generalized_coords, generalized_velocities))
    lagrange_eq = sp.simplify(ddt_dL_dqidot - dL_dqi)
    lagrange_eqs.append(lagrange_eq)

    # 慣性行列の抽出
    for j, qjddot in enumerate(generalized_accelerations):
        M[i, j] = sp.simplify(lagrange_eq.coeff(qjddot))

    # コリオリ項の抽出
    remaining_eq = lagrange_eq - sum(M[i, j] * qjddot for j, qjddot in enumerate(generalized_accelerations))
    for j, qjdot in enumerate(generalized_velocities):
        C[i, j] = sp.simplify(remaining_eq.coeff(qjdot))

    # 重力項の抽出
    G[i] = sp.simplify(remaining_eq - sum(C[i, j] * qidot for j, qidot in enumerate(generalized_velocities)))

# 結果をテキストファイルに保存
with open("lagrange_matrices.txt", "w") as f:
    f.write("### 慣性行列 M ###\n")
    f.write(sp.pretty(M, use_unicode=False) + "\n\n")
    f.write("### コリオリ項 C ###\n")
    f.write(sp.pretty(C, use_unicode=False) + "\n\n")
    f.write("### 重力項 G ###\n")
    f.write(sp.pretty(G, use_unicode=False) + "\n")

print("慣性行列、コリオリ項、重力項を 'lagrange_matrices.txt' に保存しました。")
