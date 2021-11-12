import taichi as ti

ti.init(arch=ti.cpu)



# 杨氏模量
spring_Y = 1000
# 阻尼
drag_damping = 1
dashpot_damping = 100


# 时间间隔
dt = 1e-3

# 弹簧质点的质点
particle_m = 1.0

# 弹簧质点数量
num_particles = ti.field(ti.i32, shape=())

# 最大质点个数
N = 1024

# 每个弹簧质点的位置
positions = ti.Vector.field(2, ti.f32, N)
# 每个弹簧质点的速度
vels = ti.Vector.field(2, ti.f32, N)
# 每个弹簧质点的受力
forces = ti.Vector.field(2, ti.f32, N)
# 每个弹簧的长度
rest_length = ti.field(ti.f32, shape=(N, N))
# 质点是否固定
fixed = ti.field(ti.i32, N)
# 暂停
pause = ti.field(ti.i32, shape=())


@ti.kernel
def initialized():
    n = num_particles[None]
    # 初始化每个质点的受力情况
    for i in range(n):
        forces[i] = ti.Vector([0.0, -9.8]) * particle_m
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = positions[i] - positions[j]
                d = x_ij.normalized()
                forces[i] += -spring_Y * (x_ij.norm() / rest_length[i, j] - 1) * d

                # Dashpot damping
                v_rel = (vels[i] - vels[j]).dot(d)
                forces[i] += -dashpot_damping * v_rel * d

    # 更新每个质点的位置
    for i in range(n):
        if not fixed[i]:
            a = forces[i] / particle_m
            vels[i] += a * dt
            vels[i] *= ti.exp(-dt * drag_damping)  # Drag damping
            positions[i] += vels[i] * dt
        else:
            vels[i] = ti.Vector([0.0, 0.0])

        for d in ti.static(range(2)):
            if positions[i][d] < 0:
                positions[i][d] = 0
                vels[i][d] = 0

            if positions[i][d] > 1:
                positions[i][d] = 0.9
                vels[i][d] = 0


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fix:ti.i32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    new_particle_id = num_particles[None]
    positions[new_particle_id] = [pos_x, pos_y]
    vels[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fix
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (positions[new_particle_id] - positions[i]).norm()
        connection_radius = 0.12
        if dist < connection_radius:
            # Connect the new particle with particle i
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


def main():
    gui = ti.GUI('mass spring', res=(512, 512), background_color=0xDDDDDD)

    new_particle(0.5, 0.9, True)
    new_particle(0.5, 0.8, False)
    new_particle(0.6, 0.8, False)
    new_particle(0.7, 0.8, False)
    new_particle(0.8, 0.8, False)
    new_particle(0.9, 0.8, False)
    new_particle(0.95, 0.8, False)

    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == gui.SPACE:
                pause[None] = not pause[None]

        if pause[None]:
            for i in range(10):
                initialized()

        X = positions.to_numpy()
        n = num_particles[None]

        # Draw the springs
        for i in range(n):
            for j in range(i + 1, n):
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x444444)

        # Draw the particles
        for i in range(n):
            c = 0xFF0000
            gui.circle(pos=X[i], color=c, radius=5)

        gui.show()


if __name__ == '__main__':
    main()