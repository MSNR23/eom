import sympy as sp

# 記号の定義
q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')  # 一般化座標
q1_dot, q2_dot, q3_dot, q4_dot = sp.symbols('q1_dot q2_dot q3_dot q4_dot')  # 一般化速度
q1_ddot, q2_ddot, q3_ddot, q4_ddot = sp.symbols('q1_ddot q2_ddot q3_ddot q4_ddot')  # 一般化加速度
generalized_coords = [q1, q2, q3, q4]
generalized_velocities = [q1_dot, q2_dot, q3_dot, q4_dot]
generalized_accelerations = [q1_ddot, q2_ddot, q3_ddot, q4_ddot]

# ファイルから運動方程式を読み込む
with open("lagrange_equations4dof.txt", "r") as file:
    equations = file.readlines()

# 各運動方程式を解析
lagrange_eqs = []
for line in equations:
    if line.startswith("Equation for"):
        continue  # コメント行をスキップ
    if line.strip():  # 空行でなければ
        lagrange_eqs.append(sp.sympify(line.strip()))  # SymPy式に変換

# 各運動方程式を分解
M = sp.zeros(len(generalized_coords), len(generalized_coords))  # 質量行列
C = sp.zeros(len(generalized_coords), 1)  # コリオリ項
G = sp.zeros(len(generalized_coords), 1)  # 重力項

for i, eq in enumerate(lagrange_eqs):
    # 慣性項 (加速度 q_ddot に関する項を抽出)
    for j, q_ddoti in enumerate(generalized_accelerations):
        M[i, j] = sp.simplify(eq.coeff(q_ddoti))
    
    # コリオリ項 (速度 q_dot に関する項を抽出)
    coriolis_terms = eq - sum(M[i, j] * generalized_accelerations[j] for j in range(len(generalized_coords)))
    for j, q_doti in enumerate(generalized_velocities):
        C[i] += sp.simplify(coriolis_terms.coeff(q_doti))
    
    # 重力項 (残りの定数項を抽出)
    gravity_terms = coriolis_terms - sum(C[j] * generalized_velocities[j] for j in range(len(generalized_coords)))
    G[i] = sp.simplify(gravity_terms)

# 整理してラベル付き形式で出力
def format_matrix_with_labels(matrix, prefix):
    """
    行列を 'M_ij = ...' の形式に整形して出力する
    """
    lines = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            element = sp.simplify(matrix[i, j])
            if matrix.shape[1] == 1:  # ベクトルの場合 (C, G)
                lines.append(f"{prefix}_{i+1} = {element}")
            else:  # 行列の場合 (M)
                lines.append(f"{prefix}_{i+1}{j+1} = {element}")
    return "\n".join(lines)

# ファイルに保存
with open("dynamics_matrices_with_labels.txt", "w") as file:
    file.write("Mass Matrix (M):\n")
    file.write(format_matrix_with_labels(M, "M") + "\n\n")
    file.write("Coriolis Terms (C):\n")
    file.write(format_matrix_with_labels(C, "C") + "\n\n")
    file.write("Gravity Terms (G):\n")
    file.write(format_matrix_with_labels(G, "G") + "\n\n")

# 完了メッセージ
print("Dynamics matrices with labels have been written to 'dynamics_matrices_with_labels.txt'")
