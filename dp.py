import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import csv
from matrix import compute_mass_matrix, compute_coriolis_matrix, compute_gravity_vector

# 定数
g = 9.80  # 重力加速度

# リンクの長さ
l1 = 1.0
l2 = 1.0

# 重心までの長さ
lg1 = 0.5
lg2 = 0.5

# 質点の質量
m1 = 1.0
m2 = 1.0

# 慣性モーメント
I1 = m1 * l1**2 / 3
I2 = m2 * l2**2 / 3

def log_inputs_to_csv(file_path, t, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot):
    """
    入力値をCSVファイルに書き込む
    """
    header = ['Time', 'q1', 'q2', 'q3', 'q4', 'q1_dot', 'q2_dot', 'q3_dot', 'q4_dot']
    
    # ファイルが存在しない場合はヘッダーを追加
    try:
        with open(file_path, 'r'):
            pass  # ファイルが存在する場合は何もしない
    except FileNotFoundError:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    
    # データを追記
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([t, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot])


# 運動方程式（独立関数）
def equations_of_motion(t, s, F, log=False):
    q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot = s

    # 入力値をCSVに保存
    if log:
        csv_file_path = 'input_values_log.csv'
        log_inputs_to_csv(csv_file_path, t, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot)

    
    # 質量行列
    M = compute_mass_matrix(I1, I2, m1, m2, lg1, lg2, l1, q1, q2, q3, q4, t=t)

    # コリオリ行列
    C = compute_coriolis_matrix(I1, I2, l1, lg1, lg2, m1, m2, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot)

    # 重力ベクトル
    G = compute_gravity_vector(I1, I2, l1, lg1, lg2, m1, m2, g, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot)

    # 加速度を計算
    M_inv = np.linalg.inv(M)
    q_ddot = M_inv.dot(F - C - G)

    return np.hstack((np.array([q1_dot, q2_dot, q3_dot, q4_dot]), q_ddot.ravel()))


# Runge-Kutta法
def runge_kutta(t, s, F, dt):
    k1 = dt * equations_of_motion(t, s, F, log=False)
    k2 = dt * equations_of_motion(t + 0.5 * dt, s + 0.5 * k1, F, log=False)
    k3 = dt * equations_of_motion(t + 0.5 * dt, s + 0.5 * k2, F, log=False)
    k4 = dt * equations_of_motion(t + dt, s + k3, F, log=True)

    s_new = s + (k1 + 2*k2 + 2*k3 + k4) / 6

    return s_new

# アニメーションの作成
def update_3d(frame, line, s_values):
    q1, q2, q3, q4 = s_values[frame, :4]
    
    # リンク1の先端（3次元座標）
    x1 = l1 * np.sin(q1) * np.cos(q2)
    y1 = l1 * np.sin(q1) * np.sin(q2)
    z1 = l1 * np.cos(q1)
    
    # リンク2の先端（3次元座標）
    x2 = x1 + l2 * (np.sin(q1) * np.cos(q2 + q3) * np.cos(q4) - np.cos(q1) * np.sin(q4))
    y2 = y1 + l2 * (np.sin(q1) * np.sin(q2 + q3) * np.cos(q4))
    z2 = z1 + l2 * (np.cos(q1) * np.cos(q4) + np.sin(q1) * np.sin(q4) * np.cos(q2 + q3))
    
    line.set_data([0, x1, x2], [0, y1, y2])
    line.set_3d_properties([0, z1, z2])
    return line,

# main関数
def main():
    # シミュレーションの初期化
    dt = 0.001  # 時間刻み幅
    t_end = 5.0  # シミュレーション終了時間
    t_values = np.arange(0, t_end, dt)
    s_values = np.zeros((len(t_values), 8))  # 4自由度用の状態配列

    # 初期条件
    s0 = np.array([np.pi / 4, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初期の角度、角速度
    F = np.array([0.0, 0.0, 0.0, 0.0])  # 外力（ここではゼロ）

    # シミュレーション実行
    for i, t in enumerate(t_values):
        s_values[i] = s0
        s0 = runge_kutta(t, s0, F, dt)  # 修正点: s0を直接更新する

    # データをCSVファイルに保存
    csv_file_path = 'double-pendulum_4dof_simulation_data_debug2.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Theta3', 'Theta4', 'Omega1', 'Omega2', 'Omega3', 'Omega4'])  # ヘッダー行
        for t, *state in zip(t_values, *s_values.T):
            csv_writer.writerow([t] + list(state))

    print(f'Data has been saved to {csv_file_path}')

    # アニメーションの作成
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_box_aspect([1, 1, 1])  # 均一スケール
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    line, = ax.plot([], [], [], 'o-', lw=2)

    ani = FuncAnimation(fig, update_3d, frames=len(t_values), fargs=(line, s_values), interval=50, blit=False)

    # アニメーションの保存（MP4形式）
    animation_file_path = 'double-pendulum_4dof_animation_debug2_3d.mp4'
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(animation_file_path, writer=writer)
    print(f'Animation has been saved to {animation_file_path}')

    plt.show()

if __name__ == "__main__":
    main()
