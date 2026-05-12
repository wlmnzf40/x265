# ARM SVE Optimization: LumaHVPP Vertical Filter

## 背景：算子说明

`luma_hvpp`（Luma Horizontal+Vertical Pixel-to-Pixel）是 H.265/HEVC 编码中亚像素插值的核心算子，用于对亮度分量做 8 抽头 Luma 滤波。它是运动搜索（ME）和参考帧插值（FMC）中调用最频繁的操作之一，在 `LumaHVPP32x32`、`LumaHVPP64x64` 等大块上尤为耗时。

该算子采用**二维可分离滤波**实现：

```
uint8 src  →  [水平 pass]  →  int16 immed  →  [垂直 pass]  →  uint8 dst
              (pixel-to-short)                 (short-to-pixel)
```

8 抽头滤波系数（`g_lumaFilter[4][8]`，仅用 coeffIdx 1/2/3）：

| coeffIdx | 系数 |
|----------|------|
| 1 | `{-1, 4, -10, 58, 17, -5, 1, 0}` |
| 2 | `{-1, 4, -11, 40, 40, -11, 4, -1}`（对称） |
| 3 | `{ 0, 1,  -5, 17, 58, -10, 4, -1}` |

---

## 原有 ARM 优化现状

x265 代码库中，`luma_hvpp` 已有三层 ARM 优化，按 CPU 特性检测后覆盖注册：

| 层级 | 水平 pass | 垂直 pass | 实现文件 |
|------|-----------|-----------|----------|
| NEON（基线） | `vmlal_n_s16` / `vaddl_s16` | NEON，**lo/hi 双累加器** | `filter-prim.cpp` |
| NEON DotProd | `vsdotq_lane_s32` 点积加速 | NEON，**lo/hi 双累加器** | `filter-neon-dotprod.cpp` |
| NEON i8mm | `mmla` 矩阵乘累加（最优） | NEON，**lo/hi 双累加器** | `filter-neon-i8mm.cpp` |

**关键瓶颈**：三层优化均未触及垂直 pass。垂直 pass 处理 `int16_t` 中间数据，NEON 的 32 位累加器每寄存器只有 4 个 lane；每次处理 8 列必须将 `int16x8_t` 用 `vget_low/vget_high` 拆成两个 `int32x4_t`（lo + hi），导致所有乘加指令翻倍。

---

## 本次优化：SVE 垂直滤波器

### 核心思路

利用 SVE 的两个关键指令消除 NEON 的 lo/hi 拆分开销：

- **`svld1sh_s32`**：从内存加载 `int16` 并原地符号扩展为 `int32`，装入宽度可变的 SVE 寄存器。不需要任何拆分操作。
- **`svst1b_s32`**：将每个 `int32` lane 的最低字节打包写入连续内存，直接完成 int32→uint8 的饱和截断存储。

### 可伸缩列循环

```
col loop step = svcntw()
              = 4  @ VL=128-bit (等同 NEON，但无 lo/hi 分拆)
              = 8  @ VL=256-bit (AWS Graviton4 / Neoverse V2)
              = 16 @ VL=512-bit
```

同一份二进制在不同 SVE 硬件上自动扩展吞吐量，无需重新编译。

### 指令对比（每输出行，8 列）

| | NEON | SVE VL=128 | SVE VL=256 |
|---|---|---|---|
| 加载 | `vld1q_s16` + `vget_low/high` ×2 | `svld1sh_s32` ×1 | `svld1sh_s32` ×1（处理 8 列） |
| 乘加 | ~8 条（lo+hi 各 4 tap） | ~6 条（无拆分） | ~6 条（2× 元素数） |
| 存储 | `vshrn`+`vqmovun`+`vst1` | `svst1b_s32` ×1 | `svst1b_s32` ×1 |
| 合计指令 | ~13 | ~9 | ~9，但 2× 输出 |

### 实现要点

**1. 不能用 SVE 类型数组**

`svint32_t` 大小在编译期未知，不允许指针算术（`in[3]` 非法）。
改用 11 个具名变量 `w0`…`w10` 充当滑动窗口，通过显式赋值旋转：

```cpp
// 禁止：svint32_t in[11];  in[3] 会报错
// 正确：
svint32_t w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10;
// 窗口滑动：
w0=w4; w1=w5; w2=w6; w3=w7; w4=w8; w5=w9; w6=w10;
```

**2. 非模板 SVE 辅助函数**

GCC 两阶段名称查找要求：SVE 内联函数名在模板定义时就需可见。
将三组系数的计算抽取为普通（非模板）函数：

```cpp
static inline svint32_t compute_coeff1(svbool_t pg, svint32_t r0, ..., int32_t offset);
static inline svint32_t compute_coeff2(svbool_t pg, svint32_t r0, ..., int32_t offset);
static inline svint32_t compute_coeff3(svbool_t pg, svint32_t r0, ..., int32_t offset);
static inline svint32_t saturate_sve  (svbool_t pg, svint32_t acc, int shift);
```

**3. GCC 13 SVE API 正确写法**

GCC 的 SVE 头文件不提供 `_n_` 标量广播后缀，必须用 `svdup_n_*` 显式广播：

```cpp
// 错误（ACLE 规范有，但 GCC 13 未实现）：
svmla_n_s32(pg, acc, x, 58)
svasr_n_s32(pg, acc, 12)

// 正确：
svmla_s32_x(pg, acc, x, svdup_n_s32(58))
svasr_s32_x(pg, acc, svdup_n_u32(12))
```

**4. 组合滤波器**

```
interp_hv_pp_sve<W,H>:
    interp8_horiz_ps_i8mm<W,H>  ← 复用已有最优水平 pass
    interp_vert_sp_sve<W,H>     ← 本次新增 SVE 垂直 pass
```

---

## 新增文件

| 文件 | 作用 |
|------|------|
| `source/common/aarch64/filter-prim-sve.h` | 公开声明 `setupFilterPrimitives_sve()` |
| `source/common/aarch64/filter-prim-sve.cpp` | SVE 垂直滤波器实现 |
| `source/common/aarch64/asm-primitives.cpp` | 在 SVE 特性检测分支调用 `setupFilterPrimitives_sve(p)` |
| `source/common/CMakeLists.txt` | 将 `filter-prim-sve.cpp` 加入 SVE 编译目标 |

---

## 覆盖的块大小

注册策略不是“所有 SVE 实现无条件覆盖 NEON”。实测显示，这份 SVE 垂直滤波器使用 `svld1sh_s32` 扩展到 32-bit lane 后，列循环步长是 `svcntw()`；在常见 VL=256 的机器上每次仍只处理 8 个像素，同时还要承担谓词和 SVE intrinsic 生成代码的开销。因此：

- `luma_vsp`（纯垂直）只覆盖实测优于 NEON 的小块：

```
16×4   16×8   16×12
```

- `luma_hvpp`（HV 组合）保留较多收益块，但不覆盖大块中已经更快的 NEON/i8mm 路径：

```
16×4   16×8   16×12  16×16  16×32  16×64
24×32
32×8   32×16  32×24  32×32  32×64
64×16  64×32
```

`64×64`、`64×48` 和 `48×64` 的 HV 组合继续使用现有 NEON/i8mm 实现。

---

## 编译条件

```cmake
# CMakeLists.txt 中的条件
if(CPU_HAS_SVE AND HAVE_SVE_BRIDGE)
    # filter-prim-sve.cpp 以 -march=armv8.2-a+dotprod+i8mm+sve 编译
```

运行时通过 `cpuMask & X265_CPU_SVE` 检测后激活，不影响非 SVE 平台。手动跑 testbench 时，`--cpu SVE` 现在也会带上 SVE 路径依赖的 `NEON`、`Neon_DotProd` 和 `Neon_I8MM` 标志，避免只注册 SVE-only 子集而和 `--cpu NEON` 做不完整对比。
