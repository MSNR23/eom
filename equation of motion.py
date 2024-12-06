import sympy as sp

# シンボリック変数の定義
q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')  # ロール、ピッチ、ヨー、肘の曲げ角度
lg1, lg2, l1, l2 = sp.symbols('lg1 lg2 l1 l2')  # リンク1とリンク2の重心距離、リンクの長さ

# 回転行列の定義
def rotation_matrix_roll(q1):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(q1), -sp.sin(q1)],
        [0, sp.sin(q1), sp.cos(q1)]
    ])

def rotation_matrix_pitch(q2):
    return sp.Matrix([
        [sp.cos(q2), 0, sp.sin(q2)],
        [0, 1, 0],
        [-sp.sin(q2), 0, sp.cos(q2)]
    ])

def rotation_matrix_yaw(q3):
    return sp.Matrix([
        [sp.cos(q3), -sp.sin(q3), 0],
        [sp.sin(q3), sp.cos(q3), 0],
        [0, 0, 1]
    ])

def rotation_matrix_elbow(q4):
    return sp.Matrix([
        [sp.cos(q4), 0, sp.sin(q4)],
        [0, 1, 0],
        [-sp.sin(q4), 0, sp.cos(q4)]
    ])

# 総合回転行列（肩 + 肘）
R_roll = rotation_matrix_roll(q1)
R_pitch = rotation_matrix_pitch(q2)
R_yaw = rotation_matrix_yaw(q3)
R_shoulder = R_yaw * R_pitch * R_roll
R_elbow = rotation_matrix_elbow(q4)

# 各リンクの重心位置
# リンク1の重心位置
r1 = R_shoulder * sp.Matrix([lg1, 0, 0])  # リンク1の重心

# リンク1の端点（肘の位置）
r_elbow = R_shoulder * sp.Matrix([l1, 0, 0])  # 肘の位置

# リンク2（前腕）の重心位置
r2 = r_elbow + R_shoulder * R_elbow * sp.Matrix([lg2, 0, 0])  # リンク2の重心

# 結果を表示
print("リンク1の重心位置（文字のまま）:")
sp.pprint(r1)

print("\n肘の位置（文字のまま）:")
sp.pprint(r_elbow)

print("\nリンク2（前腕）の重心位置（文字のまま）:")
sp.pprint(r2)

