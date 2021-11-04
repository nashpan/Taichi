import taichi as ti
import numpy as np
from PIL import Image
import taichi_glsl as tg


ti.init(arch=ti.gpu)


image = np.array(Image.open('edge1.png'), dtype=np.float32)
# image /= 256.0
width, height = image.shape[0:2]
image_pixels = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))
image_pixels.from_numpy(image)

pixels = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))
blur_x_pixels = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))
blur_y_pixels = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))
glow_pixels = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))


@ti.func
def color_fetch(coord):
    i = coord[0] * width
    j = coord[1] * height
    color = image_pixels[int(i), int(j)]
   # print(color[0])
    return color


@ti.func
def grab(coord, octave, offset):
    scale = pow(2, octave)
    coord += offset
    coord *= scale
    color = ti.Vector([0, 0, 0, 0])
    if coord[0] < 0 or coord[0] > 1 or coord[1] < 0 or coord[1] > 1:
        color = ti.Vector([0, 0, 0, 0])
    else:
        pixel = ti.Vector([1.0 / width, 1.0 / height]) * scale * 0.50
        off0 = ti.Vector([-2.0, -2.0]) * pixel
        off1 = ti.Vector([-2.0, -1.0]) * pixel
        off2 = ti.Vector([-2.0, 0.0]) * pixel
        off3 = ti.Vector([-2.0, 1.0]) * pixel
        off4 = ti.Vector([-2.0, 2.0]) * pixel
        color += color_fetch(coord + off0)
        color += color_fetch(coord + off1)
        color += color_fetch(coord + off2)
        color += color_fetch(coord + off3)
        color += color_fetch(coord + off4)

        off0 = ti.Vector([-1.0, -2.0]) * pixel
        off1 = ti.Vector([-1.0, -1.0]) * pixel
        off2 = ti.Vector([-1.0, 0.0]) * pixel
        off3 = ti.Vector([-1.0, 1.0]) * pixel
        off4 = ti.Vector([-1.0, 2.0]) * pixel
        color += color_fetch(coord + off0)
        color += color_fetch(coord + off1)
        color += color_fetch(coord + off2)
        color += color_fetch(coord + off3)
        color += color_fetch(coord + off4)

        off0 = ti.Vector([0.0, -2.0]) * pixel
        off1 = ti.Vector([0.0, -1.0]) * pixel
        off2 = ti.Vector([0.0, 0.0]) * pixel
        off3 = ti.Vector([0.0, 1.0]) * pixel
        off4 = ti.Vector([0.0, 2.0]) * pixel
        color += color_fetch(coord + off0)
        color += color_fetch(coord + off1)
        color += color_fetch(coord + off2)
        color += color_fetch(coord + off3)
        color += color_fetch(coord + off4)

        off0 = ti.Vector([1.0, -2.0]) * pixel
        off1 = ti.Vector([1.0, -1.0]) * pixel
        off2 = ti.Vector([1.0, 0.0]) * pixel
        off3 = ti.Vector([1.0, 1.0]) * pixel
        off4 = ti.Vector([1.0, 2.0]) * pixel
        color += color_fetch(coord + off0)
        color += color_fetch(coord + off1)
        color += color_fetch(coord + off2)
        color += color_fetch(coord + off3)
        color += color_fetch(coord + off4)

        off0 = ti.Vector([2.0, -2.0]) * pixel
        off1 = ti.Vector([2.0, -1.0]) * pixel
        off2 = ti.Vector([2.0, 0.0]) * pixel
        off3 = ti.Vector([2.0, 1.0]) * pixel
        off4 = ti.Vector([2.0, 2.0]) * pixel
        color += color_fetch(coord + off0)
        color += color_fetch(coord + off1)
        color += color_fetch(coord + off2)
        color += color_fetch(coord + off3)
        color += color_fetch(coord + off4)

    color /= 25.0
    return color / 255.0


@ti.func
def calc_offset(octave):
    offset = ti.Vector([0.0, 0.0])
    padding = ti.Vector([10.0 / width, 10.0 / height])
    offset.x = -min(1.0, ti.floor(octave / 3.0)) * (0.25 + padding.x)
    offset.y = -(1.0 - (1.0 / pow(2, octave))) - padding.y * octave
    offset.y += min(1.0, ti.floor(octave / 3.0)) * 0.35
    return offset


@ti.func
def color_fetch_pixels(uv):
    i = uv[0] * width
    j = uv[1] * height
    color = pixels[int(i), int(j)]
    return color


@ti.func
def gaussian_blur_x(uv):
    color = ti.Vector([0.0, 0.0, 0.0, 0.0])

    pixel = ti.Vector([1.0 / width, 1.0 / height])

    offset = ti.Vector([0.0, 0.0])
    color += color_fetch_pixels(uv) * 0.19638062
    color += color_fetch_pixels(uv) * 0.19638062

    offset = ti.Vector([1.41176471, 1.41176471]) * pixel
    color += color_fetch_pixels(uv + offset * ti.Vector([0.5, 0.0])) * 0.29675293
    color += color_fetch_pixels(uv - offset * ti.Vector([0.5, 0.0])) * 0.29675293

    offset = ti.Vector([3.29411765, 3.29411765]) * pixel
    color += color_fetch_pixels(uv + offset * ti.Vector([0.5, 0.0])) * 0.09442139
    color += color_fetch_pixels(uv - offset * ti.Vector([0.5, 0.0])) * 0.09442139

    offset = ti.Vector([5.17647059, 5.17647059]) * pixel
    color += color_fetch_pixels(uv + offset * ti.Vector([0.5, 0.0])) * 0.01037598
    color += color_fetch_pixels(uv - offset * ti.Vector([0.5, 0.0])) * 0.01037598

    offset = ti.Vector([7.05882353, 7.05882353]) * pixel
    color += color_fetch_pixels(uv + offset * ti.Vector([0.5, 0.0])) * 0.00025940
    color += color_fetch_pixels(uv - offset * ti.Vector([0.5, 0.0])) * 0.00025940

    color /= 1.19638064

    return color


@ti.func
def color_fetch_blur_x_pixels(uv):
    i = uv[0] * width
    j = uv[1] * height
    color = blur_x_pixels[int(i), int(j)]
    return color


@ti.func
def gaussian_blur_y(uv):
    color = ti.Vector([0.0, 0.0, 0.0, 0.0])

    pixel = ti.Vector([1.0 / width, 1.0 / height])

    offset = ti.Vector([0.0, 0.0])
    color += color_fetch_blur_x_pixels(uv) * 0.19638062
    color += color_fetch_blur_x_pixels(uv) * 0.19638062

    offset = ti.Vector([1.41176471, 1.41176471]) * pixel
    color += color_fetch_blur_x_pixels(uv + offset * ti.Vector([0.0, 0.5])) * 0.29675293
    color += color_fetch_blur_x_pixels(uv - offset * ti.Vector([0.0, 0.5])) * 0.29675293

    offset = ti.Vector([3.29411765, 3.29411765]) * pixel
    color += color_fetch_blur_x_pixels(uv + offset * ti.Vector([0.0, 0.5])) * 0.09442139
    color += color_fetch_blur_x_pixels(uv - offset * ti.Vector([0.0, 0.5])) * 0.09442139

    offset = ti.Vector([5.17647059, 5.17647059]) * pixel
    color += color_fetch_blur_x_pixels(uv + offset * ti.Vector([0.0, 0.5])) * 0.01037598
    color += color_fetch_blur_x_pixels(uv - offset * ti.Vector([0.0, 0.5])) * 0.01037598

    offset = ti.Vector([7.05882353, 7.05882353]) * pixel
    color += color_fetch_blur_x_pixels(uv + offset * ti.Vector([0.0, 0.5])) * 0.00025940
    color += color_fetch_blur_x_pixels(uv - offset * ti.Vector([0.0, 0.5])) * 0.00025940

    color /= 1.19638064

    return color


@ti.func
def cubic(x):
    x2 = x * x
    x3 = x2 * x
    w = ti.Vector([0.0, 0.0, 0.0, 0.0])
    w.x = -x3 + 3.0 * x2 - 3.0 * x + 1.0
    w.y = 3.0*x3 - 6.0 * x2 + 4.0
    w.z = -3.0*x3 + 3.0 * x2 + 3.0 * x + 1.0
    w.w = x3
    color = w / 6.0
    return color


@ti.func
def color_fetch_blur_y_pixels(uv):
    i = uv[0] * width
    j = uv[1] * height
    color = blur_y_pixels[int(i), int(j)]
    return color


@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a


@ti.func
def bicubic_texture(coord):
    coord *= ti.Vector([width, height])
    fx = coord.x - ti.floor(coord.x)
    fy = coord.y - ti.floor(coord.y)

    coord.x -= fx
    coord.y -= fy
    fx -= 0.5
    fy -= 0.5
    x_cubic = cubic(fx)
    y_cubic = cubic(fy)

    c = ti.Vector([coord.x - 0.5, coord.x + 1.5, coord.y - 0.5, coord.y + 1.5])
    s = ti.Vector([x_cubic.x + x_cubic.y, x_cubic.z + x_cubic.w, y_cubic.x + y_cubic.y, y_cubic.z + y_cubic.w])
    offset = c + ti.Vector([x_cubic.y, x_cubic.w, y_cubic.y, y_cubic.w]) / s
    sample0 = color_fetch_blur_y_pixels(ti.Vector([offset.x / width, offset.z / height]))
    sample1 = color_fetch_blur_y_pixels(ti.Vector([offset.y / width, offset.z / height]))
    sample2 = color_fetch_blur_y_pixels(ti.Vector([offset.x / width, offset.w / height]))
    sample3 = color_fetch_blur_y_pixels(ti.Vector([offset.y / width, offset.w / height]))
    sx = s.x / (s.x + s.y)
    sy = s.z / (s.z + s.w)
    color = mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy)
    return color


@ti.func
def grab_glow(coord, octave, offset):
    scale = pow(2, octave)
    coord /= scale
    coord -= offset
    color = bicubic_texture(coord)
    return color


@ti.func
def glow_calc_offset(octave):
    offset = ti.Vector([0.0, 0.0])
    padding = ti.Vector([10.0 / width, 10.0 / height])
    offset.x = -ti.min(1.0, ti.floor(octave / 3.0)) * (0.25 + padding.x)
    offset.y = -(1.0 - (1.0 / pow(2, octave))) - padding.y * octave
    offset.y += ti.min(1.0, ti.floor(octave / 3.0)) * 0.35
    return offset


@ti.func
def get_bloom(coord):
    weight = 1.0
    sum_weight = 0.0
    bloom = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for i in range(1,6):
        bloom += grab_glow(coord, float(i), glow_calc_offset(float(i) - 1.0)) * weight
        sum_weight += weight
        weight *= 0.667
    bloom /= sum_weight
    return bloom


@ti.func
def atom_rgba_2_hsla(rgba):
    hsla = ti.Vector([0.0, 0.0, 0.0, rgba.w])
    sort = rgba
    if sort.x < sort.y:
        sort.xy = sort.yx
    if sort.x < sort.z:
        sort.xz = sort.zx
    if sort.y < sort.z:
        sort.yz = sort.zy
    c_dif = sort.x - sort.z
    c_add = sort.x + sort.z
    del_c = ti.Vector([sort.x, sort.x, sort.x])
    hsla.z = c_add * 0.5
    if c_dif != 0.0:
        hsla.y = c_dif / min(c_add, 2.0 - c_add)
        del_c = (del_c - rgba.xyz + c_dif * 3.0) / (c_dif * 6.0)
        if rgba.x == sort.x:
            hsla.x = del_c.z - del_c.y
        elif rgba.y == sort.x:
            hsla.x = 0.333333 + del_c.x - del_c.z
        else:
            hsla.x = 0.666667 + del_c.y - del_c.x
        if hsla.x < 0.0:
            hsla.x += 1.0
        if hsla.x > 1.0:
            hsla.x -= 1.0

    return hsla


@ti.func
def clamp(x, a_min, a_max):
    return min(max(x, a_min), a_max)


@ti.func
def atom_hsla_2_rgba(hsla):
    rgba = hsla.zzzw
    if hsla.y != 0.0:
        q = hsla.z + hsla.y * min(hsla.z, 1.0 - hsla.z)
        p = hsla.z * 2.0 - q
        c = ti.Vector([hsla.x + 0.333333, hsla.x, hsla.x - 0.333333])
        if c.x > 1.0:
            c.x -= 1.0
        if c.z < 0.0:
            c.z += 1.0
        mu = clamp(c * 6.0, 0.0, 1.0)
        mv = clamp(4.0 - c * 6.0, 0.0, 1.0)
        vp = ti.Vector([p, p, p])
        vq = ti.Vector([q, q, q])
        vm = mix(vp, vq, mu)
        vn = mix(vp, vq, mv)
        vs = tg.scalar.step(0.5, c)
        rgba.xyz = mix(vm, vn, vs)

        return rgba


@ti.kernel
def render():
    for i, j in image_pixels:
        color = ti.Vector([0.0, 0.0, 0.0, 0.0])
        uv = ti.Vector([i / width, j / height])
        color += grab(uv, 1.0, calc_offset(0.0))
        color += grab(uv, 2.0, calc_offset(1.0))
        color += grab(uv, 3.0, calc_offset(2.0))
        color += grab(uv, 4.0, calc_offset(3.0))
        color += grab(uv, 5.0, calc_offset(4.0))
        pixels[i, j] = color


@ti.kernel
def gaussian_render_x():
    for i, j in blur_x_pixels:
        uv = ti.Vector([i / width, j / height])
        color = ti.Vector([0.0, 0.0, 0.0, 0.0])
        color += gaussian_blur_x(uv)
        blur_x_pixels[i, j] = color


@ti.kernel
def gaussian_render_y():
    for i, j in blur_y_pixels:
        uv = ti.Vector([i / width, j / height])
        color = ti.Vector([0.0, 0.0, 0.0, 0.0])
        color += gaussian_blur_y(uv)
        blur_y_pixels[i, j] = color


@ti.kernel
def glow_render():
    for i, j in glow_pixels:
        uv = ti.Vector([i / width, j / height])
        color = get_bloom(uv) * 5.0 / 10.0

        u_glow_color = ti.Vector([1.0, 0.0, 0.0, 1.0])
        color = pow(color, 1.0 / 2.2)
        color.w *= 2.0
        hsl_color = atom_rgba_2_hsla(color)
        glow_color = atom_rgba_2_hsla(u_glow_color)
        hsl_color.x = glow_color.x
        hsl_color.y = glow_color.y
        color = atom_hsla_2_rgba(hsl_color)
        color = clamp(color, 0.0, 1.0)

        glow_pixels[i, j] = color


gui = ti.GUI('taichi', res=(width, height))


for ts in range(100000):
    render()
    gaussian_render_x()
    gaussian_render_y()
   # glow_render()
    gui.set_image(blur_y_pixels.to_numpy())
    gui.show()








