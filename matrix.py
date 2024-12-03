# matrix_definitions.py
import numpy as np
import csv

def debug_matrix_to_csv(matrix, name="Matrix", t=None, filepath="mass_matrix_log.csv"):
    """
    行列をCSV形式で保存するデバッグ出力
    """
    with open(filepath, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if t is not None:
            csv_writer.writerow([f"{name} at time {t:.6f}"])
        else:
            csv_writer.writerow([f"{name}"])
        for row in matrix:
            csv_writer.writerow(row)
        csv_writer.writerow([])

def log_mass_matrix_inputs_and_result(t, q1, q2, q3, q4, M):
    """
    質量行列の計算結果をCSVに記録
    """
    file_path = 'mass_matrix_inputs_and_results.csv'
    header = ['Time','M_11', 'M_12', 'M_13', 'M_14', 'M_21', 'M_22', 'M_23', 'M_24', 'M_31', 'M_32', 'M_33', 'M_34', 'M_41', 'M_42', 'M_43', 'M_44']
    
    # ファイルが存在しない場合はヘッダーを書き込む
    try:
        with open(file_path, 'r'):
            pass
    except FileNotFoundError:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    # 入力値と行列の要素を追記
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row = [t, q1, q2, q3, q4] + M.flatten().tolist()
        writer.writerow(row)

def compute_mass_matrix(I1, I2, m1, m2, lg1, lg2, l1, q1, q2, q3, q4, t = None):
    """
    質量行列 M(q) を計算
    """
    M_11 = I1 + I2 + lg2**2 * m2 * np.sin(q4)**2
    M_12 = -lg2 * m2 * (l1 + lg2 * np.cos(q4)) * np.sin(q1) * np.sin(q4)
    M_13 = (-I1 * np.sin(q2) - I2 * np.sin(q2) + 
            l1 * lg2 * m2 * np.sin(q4) * np.cos(q1) * np.cos(q2) - 
            lg2**2 * m2 * np.sin(q2) * np.sin(q4)**2 + 
            lg2**2 * m2 * np.sin(q4) * np.cos(q1) * np.cos(q2) * np.cos(q4))
    M_14 = I2 * np.cos(q1) * np.cos(q2)

    M_21 = -lg2 * m2 * (l1 + lg2 * np.cos(q4)) * np.sin(q1) * np.sin(q4)
    M_22 = (I1 + I2 + l1**2 * m2 + 2 * l1 * lg2 * m2 * np.cos(q4) + 
            lg1**2 * m1 - lg2**2 * m2 * np.sin(q1)**2 * np.sin(q4)**2 + lg2**2 * m2)
    M_23 = (lg2 * m2 * (l1 * np.sin(q2) + lg2 * np.sin(q2) * np.cos(q4) + 
            lg2 * np.sin(q4) * np.cos(q1) * np.cos(q2)) * np.sin(q1) * np.sin(q4))
    M_24 = (-I2 * np.sin(q1) + l1 * lg2 * m2 * np.cos(q1) * np.cos(q4) + 
            lg2**2 * m2 * np.cos(q1))

    M_31 = (-I1 * np.sin(q2) - I2 * np.sin(q2) + 
            l1 * lg2 * m2 * np.sin(q4) * np.cos(q1) * np.cos(q2) - 
            lg2**2 * m2 * np.sin(q2) * np.sin(q4)**2 + 
            lg2**2 * m2 * np.sin(q4) * np.cos(q1) * np.cos(q2) * np.cos(q4))
    M_32 = (lg2 * m2 * (l1 * np.sin(q2) + lg2 * np.sin(q2) * np.cos(q4) + 
            lg2 * np.sin(q4) * np.cos(q1) * np.cos(q2)) * np.sin(q1) * np.sin(q4))
    M_33 = (I1 + I2 + l1**2 * m2 * np.cos(q2)**2 - 
            2 * l1 * lg2 * m2 * np.sin(q2) * np.sin(q4) * np.cos(q1) * np.cos(q2) + 
            2 * l1 * lg2 * m2 * np.cos(q2)**2 * np.cos(q4) + lg1**2 * m1 * np.cos(q2)**2 - 
            0.125 * lg2**2 * m2 * (-np.cos(-q1 + 2 * q2 + 2 * q4) + 
            np.cos(q1 - 2 * q2 + 2 * q4) + np.cos(q1 + 2 * q2 - 2 * q4) - 
            np.cos(q1 + 2 * q2 + 2 * q4)) + 
            lg2**2 * m2 * np.cos(q1)**2 * np.cos(q2)**2 * np.cos(q4)**2 - 
            lg2**2 * m2 * np.cos(q1)**2 * np.cos(q2)**2 + 
            lg2**2 * m2 * np.cos(q2)**2 * np.cos(q4)**2 - 
            lg2**2 * m2 * np.cos(q4)**2 + lg2**2 * m2)
    M_34 = lg2 * m2 * (l1 * np.cos(q4) + lg2) * np.sin(q1) * np.cos(q2)

    M_41 = I2 * np.cos(q1) * np.cos(q2)
    M_42 = (-I2 * np.sin(q1) + l1 * lg2 * m2 * np.cos(q1) * np.cos(q4) + 
            lg2**2 * m2 * np.cos(q1))
    M_43 = lg2 * m2 * (l1 * np.cos(q4) + lg2) * np.sin(q1) * np.cos(q2)
    M_44 = I2 + lg2**2 * m2

    M = np.array([
        [M_11, M_12, M_13, M_14],
        [M_21, M_22, M_23, M_24],
        [M_31, M_32, M_33, M_34],
        [M_41, M_42, M_43, M_44]
    ])

    # CSVに保存
    debug_matrix_to_csv(M, "Mass Matrix", t=t)
    log_mass_matrix_inputs_and_result(t, q1, q2, q3, q4, M)
    return M



def compute_coriolis_matrix(I1, I2, l1, lg1, lg2, m1, m2, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot):
    """
    コリオリ行列 C(q, q_dot) を計算
    """
    C_1 = (
        -I1 * q2_dot * np.cos(q2)
        - I1 * q3_dot * np.cos(q2)
        - I2 * q2_dot * np.sin(q2) * np.cos(q1)
        + I2 * q2_dot * np.cos(q1)
        - I2 * q2_dot * np.cos(q2)
        - I2 * q3_dot * np.cos(q2)
        - I2 * q4_dot * np.sin(q2) * np.cos(q1)
        + I2 * q4_dot * np.cos(q1)
        - 2 * l1 * lg2 * m2 * q2_dot * np.sin(q2) * np.sin(q4) * np.cos(q1)
        - 2 * l1 * lg2 * m2 * q3_dot * np.sin(q2) * np.sin(q4) * np.cos(q1)
        + 2 * lg2**2 * m2 * q1_dot * np.sin(q4) * np.cos(q4)
        + 2 * lg2**2 * m2 * q2_dot * np.sin(q1)**2 * np.sin(q4)**2 * np.cos(q2)
        + 2 * lg2**2 * m2 * q2_dot * np.sin(q1) * np.sin(q4)**2
        - 2 * lg2**2 * m2 * q2_dot * np.sin(q2) * np.sin(q4) * np.cos(q1) * np.cos(q4)
        - 2 * lg2**2 * m2 * q2_dot * np.sin(q4)**2 * np.cos(q2)
        + 2 * lg2**2 * m2 * q3_dot * np.sin(q1)**2 * np.sin(q4)**2 * np.cos(q2)
        - 2 * lg2**2 * m2 * q3_dot * np.sin(q4) * np.sin(q2 + q4) * np.cos(q1)
        - 2 * lg2**2 * m2 * q3_dot * np.sin(q4) * np.sin(q2 + q4)
        + 2 * lg2**2 * m2 * q4_dot * np.sin(q1) * np.sin(q4)**2
        - 2 * lg2**2 * m2 * q4_dot * np.sin(q2) * np.sin(q4) * np.cos(q4)
        - 2 * lg2**2 * m2 * q4_dot * np.sin(q4)**2 * np.cos(q1) * np.cos(q2)
        + 2 * lg2**2 * m2 * q4_dot * np.sin(q4) * np.cos(q4)
    )

    C_2 = (
        I1 * q1_dot * np.cos(q2)
        + I1 * q3_dot * np.cos(q2)
        + I2 * q1_dot * np.sin(q2) * np.cos(q1)
        - I2 * q1_dot * np.cos(q1)
        + I2 * q1_dot * np.cos(q2)
        + I2 * q3_dot * np.cos(q2)
        + I2 * q4_dot * np.sin(q2) * np.cos(q1)
        - I2 * q4_dot * np.cos(q1)
        - 2 * l1 * lg2 * m2 * q1_dot * np.sin(q1) * np.cos(q4)
        + 2 * l1 * lg2 * m2 * q1_dot * np.sin(q2) * np.sin(q4) * np.cos(q1)
        - 2 * l1 * lg2 * m2 * q2_dot * np.sin(q4)
        + 2 * l1 * lg2 * m2 * q3_dot * np.sin(q2) * np.sin(q1 + q4)
        + 2 * l1 * lg2 * m2 * q4_dot * np.sin(q1) * np.sin(q2) * np.cos(q4)
        - 2 * l1 * lg2 * m2 * q4_dot * np.sin(q1) * np.cos(q4)
        - 2 * l1 * lg2 * m2 * q4_dot * np.sin(q4)
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q1)**2 * np.sin(q4)**2 * np.cos(q2)
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q1) * np.sin(q4)**2 * np.cos(q1)
        + 2 * lg2**2 * m2 * q1_dot * np.sin(q1) * np.sin(q4)**2
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q1)
        + 2 * lg2**2 * m2 * q1_dot * np.sin(q2) * np.sin(q4) * np.cos(q1) * np.cos(q4)
        + 2 * lg2**2 * m2 * q1_dot * np.sin(q4)**2 * np.cos(q2)
    )

    C_3 = (
        -I1 * q1_dot * np.cos(q2)
        - I1 * q2_dot * np.cos(q2)
        - I2 * q1_dot * np.cos(q2)
        - I2 * q2_dot * np.cos(q2)
        - 2 * l1**2 * m2 * q2_dot * np.sin(q2) * np.cos(q2)
        - 2 * l1**2 * m2 * q3_dot * np.sin(q2) * np.cos(q2)
        + 2 * l1 * lg2 * m2 * q1_dot * np.sin(q1) * np.sin(q2) * np.sin(q4) * np.cos(q2)
        + 2 * l1 * lg2 * m2 * q1_dot * np.cos(q1) * np.cos(q2) * np.cos(q4)
        + 4 * l1 * lg2 * m2 * q2_dot * np.sin(q2)**2 * np.sin(q4) * np.cos(q1)
        - 4 * l1 * lg2 * m2 * q2_dot * np.sin(q2) * np.cos(q2) * np.cos(q4)
        - 2 * l1 * lg2 * m2 * q2_dot * np.sin(q4) * np.cos(q1)
        + 4 * l1 * lg2 * m2 * q3_dot * np.sin(q2)**2 * np.sin(q4) * np.cos(q1)
        + 2 * l1 * lg2 * m2 * q3_dot * np.sin(q2)**2 * np.sin(q4)
        - 4 * l1 * lg2 * m2 * q3_dot * np.sin(q2) * np.cos(q2) * np.cos(q4)
        - 2 * l1 * lg2 * m2 * q3_dot * np.sin(q2) * np.cos(q2) * np.cos(q1 + q4)
        - 2 * l1 * lg2 * m2 * q3_dot * np.sin(q4) * np.cos(q1)
        - 2 * l1 * lg2 * m2 * q3_dot * np.sin(q4)
        + 2 * l1 * lg2 * m2 * q4_dot * np.sin(q2)**2 * np.sin(q4)
        - 2 * l1 * lg2 * m2 * q4_dot * np.sin(q2) * np.cos(q1) * np.cos(q2) * np.cos(q4)
        - 2 * l1 * lg2 * m2 * q4_dot * np.sin(q4)
        + 2 * l1 * lg2 * m2 * q4_dot * np.cos(q1) * np.cos(q2) * np.cos(q4)
        - 2 * lg1**2 * m1 * q2_dot * np.sin(q2) * np.cos(q2)
        - 2 * lg1**2 * m1 * q3_dot * np.sin(q2) * np.cos(q2)
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q1)**2 * np.sin(q4)**2 * np.cos(q2)
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q1) * np.sin(q2)**2 * np.sin(q4)**2 * np.cos(q1)
        + 2 * lg2**2 * m2 * q1_dot * np.sin(q1) * np.sin(q2) * np.sin(q4) * np.cos(q2) * np.cos(q4)
        + 2 * lg2**2 * m2 * q1_dot * np.sin(q1) * np.sin(q4)**2 * np.cos(q1)
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q2) * np.sin(q4) * np.cos(q4)
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q4)**2 * np.cos(q1) * np.cos(q2)
        + 2 * lg2**2 * m2 * q1_dot * np.cos(q1) * np.cos(q2)
    )

    C_4 = (
        -I2 * q1_dot * np.sin(q2) * np.cos(q1)
        - I2 * q1_dot * np.cos(q1)
        - I2 * q2_dot * np.sin(q2) * np.cos(q1)
        - I2 * q2_dot * np.cos(q1)
        - 2 * l1 * lg2 * m2 * q3_dot * np.sin(q1) * np.sin(q2) * np.cos(q4)
        + 2 * lg2**2 * m2 * q1_dot * np.sin(q1) * np.cos(q4)**2
        - 2 * lg2**2 * m2 * q1_dot * np.sin(q1)
        - 2 * lg2**2 * m2 * q2_dot * np.sin(q1) * np.sin(q4)**2
        + 0.5 * lg2**2 * m2 * q3_dot * (np.cos(q2 - 2 * q4) - np.cos(q2 + 2 * q4))
        - 0.125 * lg2**2 * m2 * q3_dot * (
            np.cos(-2 * q1 + q2 + 2 * q4)
            - np.cos(2 * q1 - q2 + 2 * q4)
            + np.cos(2 * q1 + q2 - 2 * q4)
            - np.cos(2 * q1 + q2 + 2 * q4)
        )
        - 2 * lg2**2 * m2 * q3_dot * np.sin(q1) * np.sin(q2) * np.cos(q4)**2
        + 2 * lg2**2 * m2 * q3_dot * np.sin(q4)**2 * np.cos(q1) * np.cos(q2)
        - 0.125 * lg2 * m2 * (
            16 * l1 * q2_dot * np.sin(q1) * np.sin(q2) * np.cos(q4)
            - 4 * lg2 * q1_dot * (np.cos(q2 - 2 * q4) - np.cos(q2 + 2 * q4))
            + 16 * lg2 * q1_dot * np.cos(q1) * np.cos(q2) * np.cos(q4)**2
            - 16 * lg2 * q1_dot * np.cos(q1) * np.cos(q2)
            + lg2 * q2_dot * (
                np.cos(-2 * q1 + q2 + 2 * q4)
                - np.cos(2 * q1 - q2 + 2 * q4)
                + np.cos(2 * q1 + q2 - 2 * q4)
                - np.cos(2 * q1 + q2 + 2 * q4)
            )
            + 16 * lg2 * q2_dot * np.sin(q1) * np.sin(q2) * np.cos(q4)**2
        )
    )
    C = np.array([C_1, C_2, C_3, C_4])
    return C


def compute_gravity_vector(I1, I2, l1, lg1, lg2, m1, m2, g, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot):
    """
    重力項 G(q) を計算
    """
    G_1 = (
        I1 * q1_dot * q2_dot * np.cos(q2)
        + I1 * q1_dot * q3_dot * np.cos(q2)
        - I1 * q2_dot * q3_dot * np.cos(q2)
        + I2 * q1_dot * q2_dot * np.sin(q2) * np.cos(q1)
        - I2 * q1_dot * q2_dot * np.cos(q1)
        + I2 * q1_dot * q2_dot * np.cos(q2)
        + I2 * q1_dot * q3_dot * np.cos(q2)
        + I2 * q1_dot * q4_dot * np.sin(q2) * np.cos(q1)
        - I2 * q1_dot * q4_dot * np.cos(q1)
        - I2 * q2_dot * q3_dot * np.cos(q2)
        - I2 * q2_dot * q4_dot * np.sin(q2) * np.cos(q1)
        + I2 * q2_dot * q4_dot * np.cos(q1)
        + g * lg2 * m2 * np.sin(q1) * np.sin(q4) * np.cos(q2)
        + 2 * l1 * lg2 * m2 * q1_dot * q2_dot * np.sin(q2) * np.sin(q4) * np.cos(q1)
        + 2 * l1 * lg2 * m2 * q1_dot * q3_dot * np.sin(q2) * np.sin(q4) * np.cos(q1)
        - 2 * l1 * lg2 * m2 * q2_dot * q3_dot * np.sin(q2) * np.sin(q4) * np.cos(q1)
        - l1 * lg2 * m2 * q3_dot**2 * np.sin(q1) * np.sin(q2) * np.sin(q4) * np.cos(q2)
        - lg2**2 * m2 * q1_dot**2 * np.sin(2 * q4)
        - 2 * lg2**2 * m2 * q1_dot * q2_dot * np.sin(q1)**2 * np.sin(q4)**2 * np.cos(q2)
        - 2 * lg2**2 * m2 * q1_dot * q2_dot * np.sin(q1) * np.sin(q4)**2
        + 2 * lg2**2 * m2 * q1_dot * q2_dot * np.sin(q2) * np.sin(q4) * np.cos(q1) * np.cos(q4)
        + 2 * lg2**2 * m2 * q1_dot * q2_dot * np.sin(q4)**2 * np.cos(q2)
        - 2 * lg2**2 * m2 * q1_dot * q3_dot * np.sin(q1)**2 * np.sin(q4)**2 * np.cos(q2)
        + 2 * lg2**2 * m2 * q1_dot * q3_dot * np.sin(q4) * np.sin(q2 + q4) * np.cos(q1)
        + 2 * lg2**2 * m2 * q1_dot * q3_dot * np.sin(q4) * np.sin(q2 + q4)
        + 0.5 * lg2**2 * m2 * q1_dot * q4_dot * (np.cos(q2 - 2 * q4) - np.cos(q2 + 2 * q4))
        - 2 * lg2**2 * m2 * q1_dot * q4_dot * np.sin(q1) * np.sin(q4)**2
        + 2 * lg2**2 * m2 * q1_dot * q4_dot * np.sin(q4)**2 * np.cos(q1) * np.cos(q2)
        + lg2**2 * m2 * q2_dot**2 * np.sin(q1) * np.sin(q4)**2 * np.cos(q1)
        + 2 * lg2**2 * m2 * q2_dot * q3_dot * np.sin(q1)**2 * np.sin(q4)**2 * np.cos(q2)
        - 2 * lg2**2 * m2 * q2_dot * q3_dot * np.sin(q2) * np.sin(q4) * np.cos(q1) * np.cos(q4)
        - 2 * lg2**2 * m2 * q2_dot * q3_dot * np.sin(q4)**2 * np.cos(q2)
    )

    G_2 = (
        2 * I1 * q1_dot * q3_dot * np.cos(q2)
        - I1 * q2_dot * q3_dot * np.cos(q2)
        + 2 * I2 * q1_dot * q3_dot * np.cos(q2)
        + 2 * I2 * q1_dot * q4_dot * np.sin(q2) * np.cos(q1)
        - 2 * I2 * q1_dot * q4_dot * np.cos(q1)
        - I2 * q2_dot * q3_dot * np.cos(q2)
        - I2 * q2_dot * q4_dot * np.sin(q2) * np.cos(q1)
        + I2 * q2_dot * q4_dot * np.cos(q1)
        - g * l1 * m2 * np.cos(q2)
        - g * lg1 * m1 * np.cos(q2)
        + g * lg2 * m2 * np.sin(q2) * np.sin(q4) * np.cos(q1)
    )

    G_3 = (
        -I1 * q1_dot * q2_dot * np.cos(q2)
        + 2 * I1 * q1_dot * q3_dot * np.cos(q2)
        - I2 * q1_dot * q2_dot * np.cos(q2)
        + 2 * I2 * q1_dot * q3_dot * np.cos(q2)
        + I2 * q1_dot * q4_dot * np.sin(q2) * np.cos(q1)
        - I2 * q1_dot * q4_dot * np.cos(q1)
        - I2 * q2_dot * q4_dot * np.sin(q2) * np.cos(q1)
        + I2 * q2_dot * q4_dot * np.cos(q1)
    )

    G_4 = (
        2 * I1 * q1_dot * q3_dot * np.cos(q2)
        - I2 * q1_dot**2 * np.sin(q1) * np.cos(q2)
        - I2 * q1_dot * q2_dot * np.sin(q2) * np.cos(q1)
        - I2 * q1_dot * q2_dot * np.cos(q1)
        + 2 * I2 * q1_dot * q3_dot * np.cos(q2)
        + 2 * I2 * q1_dot * q4_dot * np.sin(q2) * np.cos(q1)
        + 2 * I2 * q2_dot * q4_dot * np.cos(q1)
    )

    # Combine into a vector
    G = np.array([G_1, G_2, G_3, G_4])
    return G