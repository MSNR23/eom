import sympy as sp

# 記号の定義
q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')  # 関節角度
q1_dot, q2_dot, q3_dot, q4_dot = sp.symbols('q1_dot q2_dot q3_dot q4_dot')  # 角速度
q1_ddot, q2_ddot, q3_ddot, q4_ddot = sp.symbols('q1_ddot q2_ddot q3_ddot q4_ddot')  # 角加速度
l1, lg1, lg2, m1, m2, g = sp.symbols('l1 lg1 lg2 m1 m2 g')  # リンク長、重心長、質量、重力加速度
I1, I2 = sp.symbols('I1 I2')  # 各リンクの慣性モーメント（シンプルな場合）

# 回転行列の定義
R_x1 = sp.Matrix([
    [1, 0, 0],
    [0, sp.cos(q1), -sp.sin(q1)],
    [0, sp.sin(q1), sp.cos(q1)]
])

R_y1 = sp.Matrix([
    [sp.cos(q2), 0, sp.sin(q2)],
    [0, 1, 0],
    [-sp.sin(q2), 0, sp.cos(q2)]
])

R_z1 = sp.Matrix([
    [sp.cos(q3), -sp.sin(q3), 0],
    [sp.sin(q3), sp.cos(q3), 0],
    [0, 0, 1]
])

R_y2 = sp.Matrix([
    [sp.cos(q4), 0, sp.sin(q4)],
    [0, 1, 0],
    [-sp.sin(q4), 0, sp.cos(q4)]
])

# 重心位置の定義
G1 = sp.Matrix([
    [lg1 * sp.cos(q2) * sp.cos(q3)],
    [lg1 * sp.sin(q3) * sp.cos(q2)],
    [-lg1 * sp.sin(q2)]
])

G2 = sp.Matrix([
    [l1 * sp.cos(q2) * sp.cos(q3) + lg2 * (-(sp.sin(q1) * sp.sin(q3) + sp.sin(q2) * sp.cos(q1) * sp.cos(q3)) * sp.sin(q4) + sp.cos(q2) * sp.cos(q3) * sp.cos(q4))],
    [l1 * sp.sin(q3) * sp.cos(q2) + lg2 * (-(-sp.sin(q1) * sp.cos(q3) + sp.sin(q2) * sp.sin(q3) * sp.cos(q1)) * sp.sin(q4) + sp.sin(q3) * sp.cos(q2) * sp.cos(q4))],
    [-l1 * sp.sin(q2) + lg2 * (-sp.sin(q2) * sp.cos(q4) - sp.sin(q4) * sp.cos(q1) * sp.cos(q2))]
])

# 重心速度の計算
G1_dot = G1.jacobian([q1, q2, q3, q4]) * sp.Matrix([q1_dot, q2_dot, q3_dot, q4_dot])
G2_dot = G2.jacobian([q1, q2, q3, q4]) * sp.Matrix([q1_dot, q2_dot, q3_dot, q4_dot])

# リンク1の角速度
omega1 = sp.Matrix([0, 0, q1_dot]) \
         + R_z1 * sp.Matrix([0, q2_dot, 0]) \
         + R_z1 * R_y1 * sp.Matrix([q3_dot, 0, 0])

# リンク2の角速度（リンク1の影響 + 肘関節の回転）
omega2 = omega1 + R_z1 * R_y1 * R_x1 * sp.Matrix([0, 0, q4_dot])

# キネティックエネルギー（並進エネルギー）
T1_trans = (1/2) * m1 * (G1_dot.T * G1_dot)[0]
T2_trans = (1/2) * m2 * (G2_dot.T * G2_dot)[0]

# キネティックエネルギー（回転エネルギー）
T1_rot = (1/2) * I1 * (omega1.T * omega1)[0]  # リンク1の回転エネルギー
T2_rot = (1/2) * I2 * (omega2.T * omega2)[0]  # リンク2の回転エネルギー

# 合計キネティックエネルギー
T = T1_trans + T2_trans + T1_rot + T2_rot

# ポテンシャルエネルギー
U1 = m1 * g * G1[2]
U2 = m2 * g * G2[2]
U = U1 + U2

# ラグランジュ関数
L = T - U

# 一般化座標、速度、加速度の定義
generalized_coords = [q1, q2, q3, q4]
generalized_velocities = [q1_dot, q2_dot, q3_dot, q4_dot]
generalized_accelerations = [q1_ddot, q2_ddot, q3_ddot, q4_ddot]

# テキストファイルに保存
with open("lagrange_equations4dof.txt", "w") as file:
    for i, (qi, qidot, qiddot) in enumerate(zip(generalized_coords, generalized_velocities, generalized_accelerations), 1):
        # ∂L/∂qi を計算
        dL_dqi = sp.diff(L, qi)
        dL_dqidot = sp.diff(L, qidot)
        ddt_dL_dqidot = sum(sp.diff(dL_dqidot, v) * a for v, a in zip(generalized_velocities, generalized_accelerations)) \
                        + sum(sp.diff(dL_dqidot, q) * qidot for q, qidot in zip(generalized_coords, generalized_velocities))
        lagrange_eq = sp.simplify(ddt_dL_dqidot - dL_dqi)

        # テキストファイルに書き込み
        file.write(f"\nEquation for q{i}:\n")
        file.write(str(lagrange_eq) + "\n")

# 書き込み完了メッセージをターミナルに出力
print("Lagrange equations have been written to 'lagrange_equations4dof.txt'")