'''
title:弹簧质点模型的布料模拟
problem: cpu  16fps   gpu  2fps   why ??????????????????????

'''

import taichi as ti

ti.init(arch=ti.cpu)

# 质点的数量
particles = ti.field(ti.i32, shape=())

# 最多粒子数量
num_particle = 81
particle_mass = 1.0    # 粒子质量

position = ti.Vector.field(2, ti.f32, num_particle)  # 粒子位置
vel = ti.Vector.field(2, ti.f32, num_particle)    # 粒子速度
force = ti.Vector.field(2, ti.f32, num_particle)   # 粒子受力

dt = 1e-4

res_length = ti.field(ti.f32, shape=(num_particle, num_particle))    # 粒子之间是否连接
particle_index = ti.Vector.field(2, ti.i32, num_particle)     # 粒子的索引

fix = ti.field(ti.f32, num_particle)       # 粒子是否固定

spring_y = ti.field(ti.f32, shape=())     # 杨氏模量
drag_damping = ti.field(ti.f32, ())      # 阻尼
dashpot_damping = ti.field(ti.f32, ())


@ti.kernel
def initialize():
    for i in range(9):  # 更新各个质点的位置
        for j in range(9):
            position[i * 9 + j].x = 0.3 + 0.05 * j
            position[i * 9 + j].y = 0.9 - 0.05 * i

            particle_index[i * 9 + j] = ti.Vector([i, j])

    # 粒子之间是否连接

    for i, j in ti.ndrange(num_particle, num_particle):
    # for i in range(num_particle):
    #     for j in range(num_particle):
        diff_x = abs(particle_index[i][0] - particle_index[j][0])
        diff_y = abs(particle_index[i][1] - particle_index[j][1])
        if diff_x <= 1 and diff_y <= 1:
            res_length[i, j] = ti.Vector([diff_x, diff_y]).norm() * 0.05
            # print(res_length[i, j])


@ti.kernel
def calculate_force():

    for i in range(num_particle):

        force[i] = ti.Vector([0.0, -9.8]) * particle_mass
        for j in range(num_particle):
            if res_length[i, j] != 0:
                diff = position[i] - position[j]
                d = diff.normalized()
                force[i] += -spring_y[None] * (diff.norm() / res_length[i, j] - 1) * d

                v_rel = (vel[i] - vel[j]).dot(d)
                force[i] += -dashpot_damping[None] * v_rel * d


@ti.kernel
def update():
    for i in range(num_particle):
        if fix[i] == 0:
            a = force[i] / particle_mass
            vel[i] += a * dt
            vel[i] *= ti.exp(-dt * drag_damping[None])
            position[i] += vel[i] * dt
        else:
            vel[i] = ti.Vector([0.0, 0.0])


def main():

    gui = ti.GUI('simple cloth simulation', res=(512, 512))

    spring_y[None] = 70
    drag_damping[None] = 1
    dashpot_damping[None] = 100
    fix[0] = 1
    fix[8] = 1

    initialize()
    while True:

        for step in range(10):
            calculate_force()
            update()

        X = position.to_numpy()
        for i in range(num_particle):
            c = 0xFFFFFF
            gui.circle(pos=X[i], color=c, radius=4)

        # Draw the springs
        for i in range(num_particle):
            for j in range(i + 1, num_particle):
                if res_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x0000FF)

        gui.show()


if __name__ == '__main__':
    main()


