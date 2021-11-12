

## 1.Taichi简介

Taichi是一款**高性能空间稀疏数据结构的计算引擎**。其涉及到的计算密集型任务全部由C++写成，而前端则选择了易于上手且灵活性强的Python。

> 一个矩阵绝大部分数值为零，**且非零元素呈不规律分布时**（？？？），则该矩阵为稀疏矩阵
>
> ![image](https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/image.141x875za5r4.png) 
>
> **为什么使用稀疏矩阵**
>
> - 压缩矩阵对象的内存台面空间（存储空间更少）
> - 加速多数机器学习程序（提高算法计算效率）

### 1.1 Taichi的特点

- **高性能**

Taichi采用**命令式编程**，由于更贴近硬件，命令式编程更容易榨取并行处理器的性能，特别是物理仿真，通常有着较高的性能需求，因此Taichi特别适合用物理模拟

- **高效率**
  
  - 高度优化的视觉计算软件系统通常用**相对接近硬件**的语言（如C++、CUDA、GLSL等）来实现。这些语言提供了编写高性能程序的可能性。不幸的是，仅仅**使用 (而不进一步优化)** 一个接近硬件的语言是不够的。视觉计算任务对于高分辨率和实时性能的需求，通常意味着只有通过复杂且有挑战性的**性能工程** (performance engineering)，才能使得一个能工作的系统达到理想的性能。同时C++，CUDA这些语言具有一定的学习难度，而Taichi在设计之初就注重通过提供**领域特定语言抽象 (domain-specific language abstractions) 与编译优化 (compiler optimizations)来**同时**达到高生产力和高性能，并满足新兴视觉计算模式的需求。
  
  > 即Taichi自动的帮我们做了数据数据计算的优化
  
  ![taichi_processing](https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/taichi_processing.png) 
  
- **可移植**

Taichi的可移植性通过**多后端**的设计来达到，包括 x64, ARM, CUDA, Metal, OpenGL compute shader等。

```python
import taichi as ti

ti.init(arch=ti.gpu)   # 选择在gpu上运行，默认为cpu，选择不同的后端，修改这一行代码即可
```



### 1.2 Taichi的适用范围

Taichi主要面向**以并行for循环+命令式编程**为主体的计算任务。

比如在图像处理、计算物理等任务中，常常需要以并行的方式遍历所有的像素（或粒子、网格、节点等），在 Taichi 中这些模式均可以表达为一个并行for循环。

> 总结来说，即Taichi适用于并行计算的需求



### 1.3 某些计算在Taichi的应用范围**之外**

1. 具有领域特性硬件 (domain-specific hardware) 的任务；
2. 粒度 (granularity) 足够粗以至于函数调用、数据传输的开销可忽略，并且有高度优化的库解决方案的任务。

具体来说

- 传统渲染任务往往有着光栅化、光线追踪硬件的支持。实时图形API，如OpenGL、DirectX、Metal、Vulkan往往已经足够适用。
- 视频编码、解码任务，常常有硬件编码、解码器的支持。
- 使用标准层 (如卷积、Batch normalization等) 的深度神经网络，常常已经被深度学习框架，如TensorFlow/PyTorch较好地解决。



### 1.4 Taichi的设计目标  

1. 简化高性能视觉计算系统的开发与部署
2. 探索新的视觉计算编程语言抽象与编译技术

秉着实用主义的设计决策，工程方面，Taichi的最高设计原则是**使得Taichi更容易地被开发者使用**，这是高度实用主义的。我们在工程细节上投入了大量精力，来提高Taichi的Python前端 (frontend) 易用性和跨平台兼容性。

Taichi的编译流程如下图所示。

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/image.1zvgbcw9b3ts.png" alt="image"  /> 

几个关键设计决策如下：

- **命令式 (imperative)**。
- **可编程宏内核 (programmable megakernels)**。
- **Python嵌入 (Embed in Python)**。
- **即时 (Just-in-time, JIT) 编译**。
- **面向数据的设计 (data-oriented design)**。

视觉计算任务常被内存带宽所限制。我们**采取了面向数据的设计**（而不是传统的面向对象的设计）。这使得我们能够更好的优化缓存利用率 (cacheline utilization) 和命中率 (cache hit rate)。更进一步，Taichi实现了数据排布与算法的解耦 (decoupling)。



#### 1.4.1 Taichi的数据结构设计

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/taichi_data_structure.1daj8kxcz8g0.png" alt="taichi_data_structure" style="zoom: 80%;" /> 



## 2.Taichi的应用

目前的程序还是需要 Python 才能运行的，有必要的话可以用 Taichi 的 **AOT 技术**去脱离 Python 运行环境。相关技术快手已经落地了，在快手移动端 App 的 “魔法表情” 里点开就可以使用。

快手魔法表情

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/taichi.gif" alt="taichi" style="zoom: 67%;" /> 

**一份Taich流体模拟代码可以同时编译成Metal和OpenGL的shader代码，因此可以运行在iOS和Android设备。**	

> 跨平台，一端多用？？？类似于移动端的Flutter，RN？
>
> **最重要的：AOT技术怎么实现**    Taichi论坛给出的答案是开发中，之后也在开源计划之中

## 3.TaiChi编程语言

编写一个Taichi程序，**主要分为三个部分：数据、计算(数据处理)、显示**

TaiChi是一个嵌入在Python中的领域特定语言(DSL)。使用TaiChi,一定要初始化

```python
import taichi as ti

# 在 GPU 上运行，自动选择后端
ti.init(arch=ti.gpu)

# 在 GPU 上运行， 使用 NVIDIA CUDA 后端
ti.init(arch=ti.cuda)
# 在 GPU 上运行， 使用 OpenGL 后端
ti.init(arch=ti.opengl)
# 在 GPU 上运行， 使用苹果 Metal 后端（仅对 OS X）有效
ti.init(arch=ti.metal)

# 在 CPU 上运行 (默认)
ti.init(arch=ti.cpu)
```

在参数 `arch=ti.gpu` 下，Taichi 将首先尝试在 CUDA 上运行。 如果你的设备不支持 CUDA，那么 Taichi 将会转到 Metal 或 OpenGL。 如果所在平台不支持 GPU （CUDA、Metal 或 OpenGL），Taichi 将默认使用 CPU 运行。

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/image.6q2h2ooxz780.png" alt="image" style="zoom: 80%;" />



### 3.0 Python作用域和Taichi作用域

添加了@ti.kernel 和@ti.func修饰符的函数将会在Taichi编译器编译执行，其它部分均有python的解释器进行解析

> 这一部分是为了说明那一部分是Taichi代码

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/image.6l7p85f2mc80.png" alt="image" style="zoom: 67%;" /> 



### 3.1 数据

#### 3.1.1 基本类型

```python
# signed integers
ti.i8,   ti.i16,   ti.i32,   it.i64       
# unsigned integers
ti.u8    ti.u16    ti.u32    it.u4
# float points
ti.f32  ti.f64                            # ti.i32 和 ti.f32 使用得最多
```

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/image.64eu3x0afkg0.png" alt="image" style="zoom: 80%;" /> 

可以在初始化的时候指定默认精度

```python
ti.init(default_fp=ti.f32)  # float = ti.f32
ti.init(default_fp=ti.f64)  # float = ti.f64
ti.init(default_fp=ti.i32)  # float = ti.i32
ti.init(default_fp=ti.i64)  # float = ti.i64
```

#### 3.1.1 类型提升

taichi选择精度更高的数据类型来存储计算后的结果

```python
i32 + f32 = f32
i32 + i64 = i64
```

#### 3.1.2 类型转换

```python
# variable = ti.cast(variable, type)

@ti.kernel
def foo():
    a = 1.7
    b = ti.cast(a, ti.i32)
    c = ti.cast(b, ti.f32)
    print("b =", b) # b = 1
    print("c =", c) # c = 1.0
```

#### 3.1.3 复合类型

使用ti.types来创造复合类型，包括向量，矩阵，结构体

```python
import taichi as ti

ti.init(arch=ti.gpu)

vec3f = ti.types.vector(3, ti.f32)
mat2f = ti.types.matrix(2, 2, ti.f32)
ray = ti.types.struct(ro=vec3f, rd=vec3f, l=ti.f32)

@ti.kernel
def foo():
    a = vec3f(0.0)
    print(a)    # [0.0, 0.0, 0.0]
    b = vec3f(0.0, 1.0, .0.0)
    print(b)    # [0.0, 1.0, 0.0]
    c = mat2f([[1.5, 1.4], [1.3, 1.2]])
   	r = ray(ro=a, rd=b, l=1)
    print("r.ro =",r.ro)   # r.ro = [0.0, 0.0, 0.0]
    print("r.rd =",r.rd)   # r.rd = [0.0, 1.0, 0.0]
```

也可以使用预定义的关键字来定义复合数据结构：ti.Vector / ti.Matrix  / ti.Struct

```python
import taichi as ti

ti.init(arch=ti.gpu)

@ti.kernel
def foo():
	a = ti.Vector([0.0, 0.0, 0.0])
    b = ti.Vector([0.0, 1.0, 0.0])
    c = ti.Matrix([[1.5, 1.4], [1.3, 1.2]])
    r = ti.Struct(v1=a, v2=b, l=1)
```

可以通过下标索引来获取复合类型的元素

```python
import taichi as ti

ti.init(arch=gpu)

@ti.kernel
def foo():
    a = ti.Vector([1.0, 2.0, 0.0])
    print(a[1])   # 2.0
    
    b = ti.Matrix([[1.5, 1.4], [1.3, 1.2]])
    print(b[1, 0])  # 1.3
```



#### 3.1.4 Fields

> Taichi is a **data**-oriented programming language where dense or spatially-sparse fields are the first-class citizens.
>
> 这个概念在Taichi里面很重要，因为我们就是以Field来构建自己的数据，这个根据一个具体的代码就可以理解了

- **Field是一个全局的元素**，在Taichi和python作用域都可以进行读写

```python
n = 320
pixels = ti.field(dtype = float, shape = (n * 2, n))   # allocates a 2D dense field named pixels of size (640, 320) and element data type float
```

- field的元素可以是标量，向量，矩阵，结构体

- 可以通过[i,j,k....]的索引来获取一个field的元素

- 特别的，获取零维场的元素使用索引[None]

```python
zero_d_scalar = ti.field(ti.f32, shape())
zero_d_scalar[None] = 1.5

zero_d_vector = ti.field(ti.f32, shape())
zero_d_vector[None] = ti.Vector([2.5, 2.6])
```

```python
# 3d gravitational field in a 256 * 256 * 128 room
gravitational_field = ti.Vector.field(n=3, dtype=ti.f32,shape=(256, 256, 128))

# 2D srain_tensor field in a 64 * 64 field
stain_tensor_field = ti.Matrix.field(n=2, m=2, dtype=ti.f32, shape=(64, 64))

# a gloabal scalar that I want to access in a Taichi kernel
global_scalar = ti.field(dtype=ti.f32, shape=())
```



##### 相关方法

```python
# a.norm(eps = 0)
# 返回:（标量）向量的大小、长度、范数

a = ti.Vector([3, 4])
a.norm() # sqrt(3*3 + 4*4 + 0) = 5           这里的0 是默认参数，a.norm(0) 

# a.norm_sqr()
# 返回:（标量）向量的大小、长度、范数的平方

a = ti.Vector([3, 4])
a.norm_sqr() # 3*3 + 4*4 = 25

# a.normalized()
#返回:（向量）向量 a 的标准化/单位向量

a = ti.Vector([3, 4])
a.normalized() # [3 / 5, 4 / 5]

# a.dot(b)
#返回:（标量） a 和 b 之间点乘（内积）的结果

a = ti.Vector([1, 3])
b = ti.Vector([2, 4])
a.dot(b) # 1*2 + 3*4 = 14

# a.cross(b)
# 返回:标量（对于输入是2维向量），或者3维向量（对于输入是3维向量）这是 a 和 b 之间叉乘的结果

a = ti.Vector([1, 2, 3])
b = ti.Vector([4, 5, 6])
c = ti.cross(a, b)   # c = [2*6 - 5*3, 4*3 - 1*6, 1*5 - 4*2] = [-3, 6, -3]

p = ti.Vector([1, 2])
q = ti.Vector([4, 5])
r = ti.cross(a, b)  # r = 1*5 - 4*2 = -3

# a.outer_product(b)
# 返回:（矩阵） a 和 b 之间张量积的结果

a = ti.Vector([1, 2])
b = ti.Vector([4, 5, 6])
c = ti.outer_product(a, b) # 注意: c[i, j] = a[i] * b[j]  # c = [[1*4, 1*5, 1*6], [2*4, 2*5, 2*6]

# a.cast(dt)
# 返回:（向量）将向量 a 中分量的数据类型转化为类型 dt

# Taichi 作用域
a = ti.Vector([1.6, 2.3])
a.cast(ti.i32) # [2, 3]
```

##### 元数据

```python
# a.n
# 返回:（标量）返回向量 a 的维度

# Taichi 作用域
a = ti.Vector([1, 2, 3])
a.n  # 3
```



### 3.2 计算

#### 3.2.1 kernels

> Taichi **kernels** are defined with the decorator `@ti.kernel`. They can be called from Python to perform computation. 

```python
import taichi as ti

ti.init(arch=ti.gpu)

@ti.kernel
def foo():
    print("foo")
    bar()   # 错误   kernel不能调用其它的kernel
    
@ti.kernel
def bar():
    print("bar")
     
foo()
```



##### 3.2.1.1 **kernel中的循环结构**

Taichi kernel最外层的for循环会自动进行并行计算,循环有两种形式，rang-for循环和strurct-for循环。

```python
@ti.kernel
def fill():
    for i in range(10): # Parallelized
        x[i] += i
        s = 0
        for j in range(5):   # serialized
            s += j
        y[i] = s
    
    for k in range(20):  # Parallelized
        z[k] = k
        
@ti.kernel
def fill_3d():
    # Parallelized for all 3 <= i < 8, 1 <= j < 6, 0<= k < 9
    for i, j, k in ti.ndrange((3,8), (1,6), (0,9)):
        x[i,j,k] = i + j + k;             
```

注意，处于作用域最外层的循环才会进行并行处理，不是作用域外层的循环不会进行并行处理。

```python
@ti.kernel
def foo():
    for i in range(10): # Parallelized 
        ...
        
@ti.kernel
def bar(k: ti.i32):
    if k > 42:
        for i in range(10)：  # Serial
        	...
```

可以用到的一些小技巧：

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/image.23qoizjqslq8.png" alt="image" style="zoom:80%;" /> 

##### 3.2.1.2 **注意**

- 在并行循环里面，break是不支持的

```python
@ti.kernel
def foo():
    for i in range(10):
        ''''''
        break     # error,因为这个是并行处理的，break不起作用
```

- 全局变量的处理

  在kernel里面使用+=进行原子的操作，

```python
@ti.kernel
total[None] = 0
def foo():
	for i in range(10):
        total[None] += 10  # 进行原子的加法，
        
foo()
print(total[None])  # 100

total2[None] = 0
@ti.kernel
def sum():
    for i in range(10):
        total2[None] = total2[None] + 10   # data race,因为这个for循环是并行处理，会进行读写操作，类比于多线程，因此最后的结果不确定
```



##### 3.2.1.3 kernel参数

kernel中的参数必须显示指明，而且**只允许传入标量**

```python
import taichi as ti

ti.init(arch=ti.cpu)

@ti.kernel
def my_kernel(x:ti.i32, y:ti.i32)   # 必须显示指定，因为参数来自于python，python是弱类型语言，taichi无法识别python的类型
	print(x + y) 
```

##### 3.2.1.4 **值传递**

```python
@ti.kernel
def foo(x:ti.i32):
    x = x + 1
    print(x)
    
x = 100
foo(x) 
print("outside x is :", x)   # 100
```



##### 3.2.1.5 Return value

一个kernel函数可能有一个标量(scalar)作为返回值或者没有，如果有返回值，返回的类型必须显示指明

```python
@ti.kernel
def my_kernel() -> ti.f32:
	return 233.33

print(my_kernel())   #233.33
```

目前来说，一个kernel函数只能有一个标量(scalar)作为返回值，返回ti.Martrix或者ti.Vector是不支持的。Python里的元祖也是不支持的。



##### 3.2.1.6 Advanced arguments

在taichi的kernel中支持模板参数和数组参数，需要使用ti.template()和ti.ext_arr()显式的指明其数据类型。

> 注意，在differentiable programming中，不要使用kernel返回值，因为返回值不会被automic differentiable捕获，替代的方法是，你可以将结果存储在全局变量中。



#### 3.2.2 function

被@ti.func修饰的函数是taichi的函数

taichi的函数只允许在taichi作用域被调用，**由于Taichi中函数是inline的，因此不能有递归**

```python
@ti.func    # 函数的作用是为了代码的复用
def add(x, y):
    return x + y

a = add(4, 5)  # error, can't be called outside Taichi scope
```

taichi的函数不用指明参数，因为taichi的函数只能被taichi的kernel或者func调用，仍在taichi作用域，因此无需指明参数类型，返回类型同理也没有太多的限制

### 3.3 显示

#### 3.4.1 GUI系统

Taichi具有内置的GUI系统，可帮助用户可视化

##### 3.4.1.1 创建窗口

```python
# 创建一个窗口
ti.GUI(title, res, bgcolor=0x000000)
       # title   :   窗口名称
       # res     :   窗口分辨率
       # bgcolor ：  背景颜色
    
# 显示
gui = ti.GUI('demo', (640, 640))
gui.show(filename=None)
       # gui     :   窗口对象
       # filename:   可选
      
    # 举个例子
    for frame in range(10000):
        render(img)
        gui.set_img(img)
        gui.show(f'{frame:06d}.png')       #  屏幕截图将会以xxxxxx.png的格式保存在frame文件夹下

# 传入数据
gui.set_image(img)
       # gui   :   窗口对象
       # img   :   (numpy数组或Taichi张量)包含图像的张量
```

一般程序中的例子：创建，传入数据，显示

```python
gui = ti.GUI("saber", res=(Width, Height)) 
for ts in range(1000000):                                         # 显示部分     
    render()     
    gui.set_image(pixels.to_numpy())     
    gui.show()
```



##### 3.4.1.2  绘制

```python
# 画一个圆
gui.circle(pos, color=0xFFFFFF, radius=1)
          # pos     :   (2元祖)圆的位置
          # color   :   RGB十六进制，颜色填充
          # radius  :   圆的半径
            
# 画多个圆
gui.circles(pos, color=0xFFFFFF, radius=1)
         # pos      :   (numpy数组)一系列圆的位置

# 画一条线
gui.line(begin, end, color=0xFFFFFF, radius=1)
         # begin    :    (2元祖)直线的第一个端点位置
         # end      :    (2元祖)直线的第二个端点位置
         # radius   :     线宽
            
# 画一条线 
gui.line(begin, end, color=0xFFFFFF, radius=1)          
        # begin    :    (numpy 数组)直线的第一个端点组成的数组         
        # end      :    (numpy 数组)直线的第二个端点组成的数组          
        # radius   :     线宽  
        
        
# 画一个三角形
gui.triangle(a, b, c, color=0xFFFFFF)
        # a        :    (2元组）三角形的第一个端点位置
        # b        :   （2元组）三角形的第二个端点位置
        # c        :   （2元组）三角形的第三个端点位置
        
# 画多个三角形  
gui.triangles(a, b, c, color=0xFFFFFF)


# 画一个空心矩形
gui.rect(topleft, bottomright, radius=1, color=0xFFFFFF)
       # topleft     :    (2元祖)矩形的左上角位置
       # bottomright :    (2元祖)矩形的右下角位置
        
        
# 文字
gui.text(content, pos, font_size=15, color=0xFFFFFF)
       # content    :    (字符串)需要绘制的文字
       # pos        :    (2元祖)字体/文字的左上角位置
       # font_size  :     字体的大小(以高度计)
```

##### 3.4.1.3  事件处理

每个事件都有key和type

type是事件的类型，目前有如下三种事件

```python
ti.GUI.RELEASE  # 键盘或鼠标被放开
ti.GUI.PRESS    # 键盘或鼠标被按下
ti.GUI.MOTION   # 鼠标移动或鼠标滚轮 
```

*事件的 ``key``* 是你在键盘或鼠标上按下的按钮，可以是以下之一：

```python
# for ti.GUI.PRESS and ti.GUI.RELEASE event:
ti.GUI.ESCAPE  # Esc
ti.GUI.SHIFT   # Shift
ti.GUI.LEFT    # Left Arrow
'a'            # we use lowercase for alphabet
'b'
...
ti.GUI.LMB     # Left Mouse Button
ti.GUI.RMB     # Right Mouse Button

# for ti.GUI.MOTION event:
ti.GUI.MOVE    # Mouse Moved
ti.GUI.WHEEL   # Mouse Wheel Scrolling
```

*事件过滤器* 是一个由 `key`，`type` 和 `(type, key)` 元组组成的列表，例如：

```python
# 如果按下或释放ESC:
gui.get_event(ti.GUI.ESCAPE)

# 如果按下任何键:
gui.get_event(ti.GUI.PRESS)

# 如果按ESC或释放SPACE:
gui.get_event((ti.GUI.PRESS, ti.GUI.ESCAPE), (ti.GUI.RELEASE, ti.GUI.SPACE))
```

#### 3.4.2 其它

其它方面类型图片的输入/输出等，以及一些界面的按钮，滑块等由于用得不多，暂时没有列出来



### 3.4 其它

#### 3.4.1 Interacting with other Python packages

##### 3.4.1.1 python-scope data access

所有在Taichi作用域外的(ti.func和ti.kernel)只是简单的Python代码。在Python作用域内，你可以使用普通的索引语法来获取TaiChi的file elements.例如，你可以在pyhton作用域通过以下的方式来获取一个单独的渲染pixel

```python
import taichi as ti
pixels = ti.field(ti.f32, (1024, 512))

pixels[42, 11] = 0.7    # store data into pixels
print(pixels[42, 11])   # prints 0.7
```

##### 3.4.1.2 Sharing data with other packages

Taichi提供了功能函数例如from_numpy和to_numpy用于Taichi fields和Numpy数组之间数据的转换，因此你可以使用混合使用numpy， pytorch, matplotlib和Taichi

```python
import taichi as ti
pixels = ti.field(ti.f32, (1024, 512)
                  
import numpy as np
arr = np.random.rand(1024, 512)
pixels.from_numpy(arr)   # load numpy data into taichi fields
                  
import matplotlib.pyplot as plt
arr = pixels.to_numpy() # store taichi data into numpy arrays
plt.imshow(arr)
plt.show()
                  
import matplotlib.cm as cm
cmap = cm.get_camp('maga')
gui = ti.GUI('Color map')
while gui.running:
	render_pixels()
	arr = pixels.to_numpy()
	gui.set_image(cmap(arr))
	gui.show()          
```





#### 3.4.2 Interacting with external arrays

尽管Taichi的场主要在Taichi的作用域中使用，但是在一些情况下，在python的作用域中有效地使用Taichi的场将会很有用。其中最典型的例子就是在Taichi和python的Numpy数组之间进行数据的交换。我们通过to_numpy(）将Taichi场中的数据导入到Numpy数组中。

```python
@ti.kernel
def my_kernel():
    for i in x:
        x[i] = i * 2

x = ti.field(ti.f32, 4)
my_kernel()
x_np = x.to_numpy()
print(x_np)      # 2,4,6,8
```

通过使用from_numpy()将Numpy中的数据导入到Taichi场中。我们也可以通过Numpy arrays来初始化Taichi的场

```python
x = ti.field(ti.f32, 4)
x_np = np.array([1, 7, 3, 5])
x.from_np(x_np)
print(x[0])
print(x[1])
print(x[2])
print(x[3])
```

##### 3.4.2.1 External array shapes

- 对于标量，NumPy和Taichi场的shape是一样的

```python
field = ti.field(ti.i32, shape=(233, 666))
field.shape  # (233, 666)

array = field.to_numpy()
array.shape  # (233, 666)

field.from_numpy(array)  # the input array must be of shape (233, 666)
```

- 对于向量场，如果向量场是n维的，那么NumPy的shape应该是`(*field_shape, vector_n)`:

```python
field = ti.Vector.field(3, ti.i32, shape=(233, 666))
field.shape  # (233, 666)
field.n      # 3

array = field.to_numpy()
array.shape  # (233, 666, 3)

field.from_numpy(array)  # the input array must be of shape (233, 666, 3)
```

- 对于矩阵场，如果矩阵是n * m，那么NumPy的shape应该是`(*field_shape, matrix_n, matrix_m)`:

```python
field = ti.Matrix.field(3, 4, ti.i32, shape=(233, 666))
field.shape  # (233, 666)
field.n      # 3
field.m      # 4

array = field.to_numpy()
array.shape  # (233, 666, 3, 4)

field.from_numpy(array)  # the input array must be of shape (233, 666, 3, 4)
```

- 对于结构体的场，外部数组将会通过字典数组的方式，通过键值对进行导入。

```python
field = ti.Struct.field({'a': ti.i32, 'b': ti.types.vector(float, 3)} shape=(233, 666))
field.shape # (233, 666)

array_dict = field.to_numpy()
array_dict.keys() # dict_keys(['a', 'b'])
array_dict['a'].shape # (233, 666)
array_dict['b'].shape # (233, 666, 3)

field.from_numpy(array_dict) # the input array must have the same keys as the field
```

##### 3.4.2.2 使用外部数组作为Taichi kernel的参数

使用显示类型指示ti.ext_arr()来传递外部数组作为kernel的参数。

```python
import taichi as ti
import numpy as np

ti.init()

n = 4
m = 7

val = ti.field(ti.i32, shape=(n, m))

@ti.kernel
def test_numpy(arr: ti.ext_arr()):
  for i in range(n):
    for j in range(m):
      arr[i, j] += i + j

a = np.empty(shape=(n, m), dtype=np.int32)

for i in range(n):
  for j in range(m):
    a[i, j] = i * j

test_numpy(a)

for i in range(n):
  for j in range(m):
    assert a[i, j] == i * j + i + j
```



#### 3.4.3 元编程

元编程可以

- **统一依赖维度的代码开发工作**，例如2维/3维的物理仿真
- 通过将运行时开销转移到编译时来提高运行时的性能
- 简化 Taichi 标准库的开发

模板元编程

你可以使用 `ti.template()` 作为类型提示来传递一个张量作为参数。例如:

```python
@ti.kernel
def copy(x: ti.template(), y: ti.template()):
    for i in x:
        y[i] = x[i]

a = ti.var(ti.f32, 4)
b = ti.var(ti.f32, 4)
c = ti.var(ti.f32, 12)
d = ti.var(ti.f32, 12)
copy(a, b)
copy(c, d)
```

使用组合索引（grouped indices）的对维度不依赖的编程

然而，上面提供的 `copy` 模板函数并不完美。例如，它只能用于复制1维张量。如果我们想复制2维张量呢？那我们需要再写一个内核吗?

```python
@ti.kernel
def copy2d(x: ti.template(), y: ti.template()):
    for i, j in x:
        y[i, j] = x[i, j]
```

没有必要！Taichi 提供了 `ti.grouped` 语法，使你可以将 for 循环索引打包成一个分组向量，以统一不同维度的内核。例如:

```python
@ti.kernel
def copy(x: ti.template(), y: ti.template()):
    for I in ti.grouped(y):
        # I is a vector with same dimensionality with x and data type i32
        # If y is 0D, then I = ti.Vector([]), which is equivalent to `None` when used in x[I]
        # If y is 1D, then I = ti.Vector([i])
        # If y is 2D, then I = ti.Vector([i, j])
        # If y is 3D, then I = ti.Vector([i, j, k])
        # ...
        x[I] = y[I]

@ti.kernel
def array_op(x: ti.template(), y: ti.template()):
    # if tensor x is 2D:
    for I in ti.grouped(x): # I is simply a 2D vector with data type i32
        y[I + ti.Vector([0, 1])] = I[0] + I[1]

    # then it is equivalent to:
    for i, j in x:
        y[i, j + 1] = i + j
```

#### 3.4.4 坐标偏移

- Taichi 张量支持 **坐标偏移(coordinate offsets)** 的定义方式。偏移量会移动张量的边界，使得张量的原点不再是零向量。一个典型的例子是在物理模拟中支持负坐标的体素。
- 例如，一个大小为 `32x64` 、起始元素坐标偏移为 `(-16, 8)` 的矩阵可以按照以下形式来定义：

```python
a = ti.Matrix(2, 2, dt=ti.f32, shape=(32, 64), offset=(-16, 8))
```

通过这样，张量的下标就是从 `(-16, 8)` 到 `(16, 72)` 了（半开半闭区间）.

```python
a[-16, 32]  # 左下角
a[16, 32]   # 右下角
a[-16, 64]  # 左上角
a[16, 64]   # 右上角
```

#### 3.4.5 面向数据对象式编程

Taichi是一种 [面向数据的](https://en.wikipedia.org/wiki/Data-oriented_design) 编程(DOP)语言。但是，单纯的DOP会使模块化变得困难。为了允许代码模块化，Taichi从面向对象编程(OOP)中借鉴了一些概念。为了方便起见，我们将称此混合方案为 **面向数据对象式编程** (ODOP)。

```python
import taichi as ti

ti.init()

@ti.data_oriented
class Array2D:
  def __init__(self, n, m, increment):
    self.n = n
    self.m = m
    self.val = ti.var(ti.f32)
    self.total = ti.var(ti.f32)
    self.increment = increment
    ti.root.dense(ti.ij, (self.n, self.m)).place(self.val)
    ti.root.place(self.total)

  @staticmethod
  @ti.func
  def clamp(x):  # Clamp to [0, 1)
      return max(0, min(1 - 1e-6, x))

  @ti.kernel
  def inc(self):
    for i, j in self.val:
      ti.atomic_add(self.val[i, j], self.increment)

  @ti.kernel
  def inc2(self, increment: ti.i32):
    for i, j in self.val:
      ti.atomic_add(self.val[i, j], increment)

  @ti.kernel
  def reduce(self):
    for i, j in self.val:
      ti.atomic_add(self.total, self.val[i, j] * 4)

arr = Array2D(128, 128, 3)

double_total = ti.var(ti.f32, shape=())

ti.root.lazy_grad()

arr.inc()
arr.inc.grad()
assert arr.val[3, 4] == 3
arr.inc2(4)
assert arr.val[3, 4] == 7

with ti.Tape(loss=arr.total):
  arr.reduce()

for i in range(arr.n):
  for j in range(arr.m):
    assert arr.val.grad[i, j] == 4

@ti.kernel
def double():
  double_total[None] = 2 * arr.total

with ti.Tape(loss=double_total):
  arr.reduce()
  double()

for i in range(arr.n):
  for j in range(arr.m):
    assert arr.val.grad[i, j] == 8
```



#### 3.4.6 语法糖

##### 3.4.6.1 别名

```python
@ti.kernel
def my_kernel():
  for i, j in tensor_a:
    tensor_b[i, j] = some_function(tensor_a[i, j])
```

张量和函数使用 `ti.static` 别名为新名称：

```python
@ti.kernel
def my_kernel():
  a, b, fun = ti.static(tensor_a, tensor_b, some_function)
  for i,j in a:
    b[i,j] = fun(a[i,j])
```

还可以为类成员和方法创建别名，这有助于防止含有 `self` 的面向对象编程代码混乱。

例如，考虑使用类内核来计算某个张量的二维拉普拉斯算子：

```python
@ti.kernel
def compute_laplacian(self):
  for i, j in a:
    self.b[i, j] = (self.a[i + 1,j] - 2.0*self.a[i, j] + self.a[i-1, j])/(self.dx**2) \
                + (self.a[i,j + 1] - 2.0*self.a[i, j] + self.a[i, j-1])/(self.dy**2)
```

使用 `ti.static()` ，这可以简化为：

```
@ti.kernel
def compute_laplacian(self):
  a,b,dx,dy = ti.static(self.a,self.b,self.dx,self.dy)
  for i,j in a:
    b[i,j] = (a[i+1, j] - 2.0*a[i, j] + a[i-1, j])/(dx**2) \
           + (a[i, j+1] - 2.0*a[i, j] + a[i, j-1])/(dy**2)
```



### 3.5 调试

这一方面目前使用最基本的方法，使用print，打印关键信息到控制台。目前，Taichi 作用域的 print 支持字符串、标量、矢量和矩阵表达式作为参数。另外也可以使用assert来帮忙



## 4.Demo

编写Taichi代码基本按照上面说的三个步骤就行，即数据，计算，显示

### 4. 1. render

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/image.6sercclsrmg0.png" alt="image" style="zoom:50%;" />     <img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/down_sample.3tr225swhoi0.png" alt="down_sample" style="zoom:65%;" />

​     shadertoy效果(perlin_noise的实时计算)                            多重降采样+横向纵向的高斯模糊    

```python
import taichi as ti

ti.init(arch=ti.gpu)

Width = 646
Height = 846
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(Width, Height))       # 数据部分

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
def render():                                                         # 数据处理部分
    for i, j in pixels:
        uv = ti.Vector([ (i + 0.0) / Width, (j + 0.0) / Height])
        uv = uv * 2.0 - 1
        uv1 = uv
        p = ti.Vector([(i + 0.0) / Width, (j + 0.0) / Width])

        intensity = noise_sum_abs_sin(p)
        intensity1 = noise_sum_abs(p)

        t = clamp(uv[1] * (-uv[1]) * 0.06 + 0.15, 0, 1)
        y = abs(-t + uv[0] - intensity * 0.15)
        # y = abs(uv[1] * uv[1] + uv[0] * uv[0] - 0.65)
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

gui = ti.GUI("saber", res=(Width, Height))                           # 显示部分

for ts in range(1000000):                                           
    render()
    gui.set_image(pixels.to_numpy())
    gui.show()
```



### 4.2. 物理模拟

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/N_body.tijhdgbm0kw.gif" alt="N_body" style="zoom:53%;" />       <img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/mass_spring.1w0iqkr4kh0g.gif" alt="mass_spring" style="zoom: 80%;" />

​         3000个粒子相互作用(粒子两两之间受万有引力作用)                                         简单的弹簧质点模型(显式实现)

说明：粒子相互作用这个例子在GPU上运行，能稳定在60fps，CPU上是7fps左右，切换只需要改变一行代码 ti.init(arch=ti.cpu),cpu上明显不够流畅

<img src="https://raw.githubusercontent.com/nashpan/image-hosting/main/markdown_image/n_body.30vysp1066g0.gif" alt="n_body" style="zoom:50%;" /> 

N_Body问题

```python
# Authored by Tiantian Liu, Taichi Graphics.
import math

import taichi as ti

ti.init(arch=ti.gpu)

# global control
paused = ti.field(ti.i32, ())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1

# number of planets
N = 3000
# unit mass
m = 1
# galaxy size
galaxy_size = 0.4
# planet radius (for rendering)
planet_radius = 2
# init vel
init_vel = 120

# time-step size
h = 1e-4
# substepping
substepping = 30

# center of the screen
center = ti.Vector.field(2, ti.f32, ())

# pos, vel and force of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)


@ti.kernel
def initialize():
    center[None] = [0.5, 0.5]
    for i in range(N):
        theta = ti.random() * 2 * math.pi
        r = (ti.sqrt(ti.random()) * 0.6 + 0.4) * galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center[None] + offset
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel


@ti.kernel
def compute_force():

    # clear force
    for i in range(N):
        force[i] = [0.0, 0.0]

    # compute gravitational force
    for i in range(N):
        p = pos[i]
        for j in range(N):
            if i != j:  # double the computation for a better memory footprint and load balance
                diff = p - pos[j]
                r = diff.norm(1e-5)

                # gravitational force -(GMm / r^2) * (diff/r) for i
                f = -G * m * m * (1.0 / r)**3 * diff

                # assign to each particle
                force[i] += f


@ti.kernel
def update():
    dt = h / substepping
    for i in range(N):
        #symplectic euler
        vel[i] += dt * force[i] / m
        pos[i] += dt * vel[i]


gui = ti.GUI('N-body problem', (800, 800))

initialize()
while gui.running:

    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == ti.GUI.SPACE:
            paused[None] = not paused[None]

    if not paused[None]:
        for i in range(substepping):
            compute_force()
            update()

    gui.circles(pos.to_numpy(), color=0xffffff, radius=planet_radius)
    gui.show()
```

## 5.To do next

语言层面

- [ ] 使用Taichi模拟流体，体积云等，进一步熟悉Taichi

应用方面

- [ ] AOT技术，即脱离python环境

> [Export Taichi kernels to C source](https://docs.taichi.graphics/zh-Hans/lang/articles/misc/export_kernels)

技术方面

- [ ] Taichi多后端技术
- [ ] Taichi内置算法的优化，理解Taichi编译优化的思想

​         

## 参考资料

1.[如何在电脑屏幕里重现各种物理现象？浅谈物理引擎的技术和生产力工具](https://www.bilibili.com/video/BV19K4y1P7fb?share_source=copy_web)

2.[taichi中文文档](https://taichi.readthedocs.io/zh_CN/latest/vector.html) 

> 说明，这个中文文档已经比较老了，新的Taichi版本语法上发生了一些变化，新版本的Taichi更强调field的概念，如需要查阅文档，建议直接食用官方的英文文档，是和最新的同步的

3.[多项核心技术进展发布！胡渊鸣创业后首现身，讲述「太极图形」的前世今生](https://www.yanxishe.com/blogDetail/27493)

4.The Taichi High-Performance and Differentiable Programming Language for Sparse and Quantized Visual Computing(Taichi作者论文)

