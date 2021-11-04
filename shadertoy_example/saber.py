import taichi as ti
import numpy as np
import matplotlib.pyplot as plt


ti.init(arch=ti.gpu)

Width = 646
Height = 846
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(Width, Height))

# pixels = ti.imread('edge1.png')


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def hash22(p):
    m = p.dot(ti.Vector([127.1, 311.7]))
    n = p.dot(ti.Vector([269.5, 183.3]))
    h = ti.Vector([m, n])

    return 2.0 * fract(ti.sin(h) * 43758.5453123) - 1.0


@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a


@ti.func
def clamp(x, a_min, a_max):
    return min(max(x, a_min), a_max)


@ti.func
def noise(p):
    pi = ti.floor(p)
    pf = p - pi

    w = pf * pf * (3.0 - 2.0 * pf)

    v1 = mix(hash22((pi + ti.Vector([0.0, 0.0]))).dot(pf - ti.Vector([0.0, 0.0])),
             hash22((pi + ti.Vector([1.0, 0.0]))).dot(pf - ti.Vector([1.0, 0.0])), w[0])
    v2 = mix(hash22((pi + ti.Vector([0.0, 1.0]))).dot(pf - ti.Vector([0.0, 1.0])),
             hash22((pi + ti.Vector([1.0, 1.0]))).dot(pf - ti.Vector([1.0, 1.0])), w[0])
    v = mix(v1, v2, w[1])

    return v


@ti.func
def noise_sum_abs(p):
    f = 0.0
    p = p * 38.0
    amp = 1.0
    for _ in range(5):
        f += amp * ti.abs(noise(p))
        p = 2.0 * p;
        amp = amp / 2.0;

    return f


@ti.func
def noise_sum_abs_sin(p):
    f = 0.0
    p = p * 7.0
    amp = 1.0
    for i in range(0, 5):
        f += amp * noise(p)
        p = 2.0 * p
        amp /= 2.0
    f = ti.sin(f + p[0] / 15.0)

    return f


@ti.kernel
def render():
    for i, j in pixels:
        uv = ti.Vector([ (i + 0.0) / Width, (j + 0.0) / Height])
        uv = uv * 2.0 - 1
        uv1 = uv
        p = ti.Vector([(i + 0.0) / Width, (j + 0.0) / Width])

        intensity = noise_sum_abs_sin(p)
        intensity1 = noise_sum_abs(p)

        t = clamp(uv[1] * (-uv[1]) * 0.06 + 0.15, 0, 1)
        y = abs(-t + uv[0] - intensity * 0.15)
        g = pow(y, 0.40)
        col = ti.Vector([1.2, 1.46, 1.6]) / 1.2
        col = col * (1 - g)
        col = col * col
        col = col * col
        col = col * col

        uv1[0] = uv1[0] + 0.1
        y1 = abs(-t + uv1[0] - intensity1 * 0.6)
        g1 = pow(y1, 0.3)

        col1 = ti.Vector([1.2, 1.46, 1.6]) / 1.2
        col1 = col1 * (1 - g1)
        col1 = col1 * col1
        col1 = col1 * col1
        col1 = col1 * col1

        final_col = col + col1
        pixels[i, j] = final_col


gui = ti.GUI("saber", res=(Width, Height))

for ts in range(1000000):
    render()
    gui.set_image(pixels.to_numpy())
    gui.show()


# ti.imshow(pixels, 'edge')
