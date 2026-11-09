# LightBarExtractor 灯条提取与匹配

> 文件：`detect/src/LightBarExtractor.cpp` / `detect/include/LightBarExtractor.hpp`。
> 就 3 个函数，只有 `extractArmor` 是公开入口，另外两个在文件内部匿名 namespace。

## 1. 三个函数的功能

| 函数 | 作用 |
|---|---|
| `refineEndpoint(gray, endpoint, dir_out)` | 灰度梯度端点精修：在二值化端点邻域沿灯条轴取灰度剖面，找梯度最大点 + 抛物线拟合到亚像素，去掉阈值切坡的抖动 |
| `findLights(roi, detect_color)` | 找灯条：二值化 → 形态学 → 轮廓筛选 → fitLine 中轴线 → 端点（投影极值 + 灰度梯度精修）→ 颜色分类，返回 `vector<Light>` |
| `extractArmor(bgr, box, detect_color)` | 对外唯一入口：ROI 裁剪 → 调 findLights → 颜色过滤 → 配对 → 输出 `ArmorResult`（4 角点 + size + 左右灯条） |

## 2. 调用顺序（从外到内）

```
armor_detect（StudentTasks.cpp）
   │  对每个 YOLO 检测框调用
   ▼
extractArmor(bgr, box, detect_color)          // 入口
   ├─ ① ROI 裁剪（外扩 10%）
   ├─ ② 找单个灯条 调用 findLights(roi, detect_color)
   │      ├─ 灰度二值化（按颜色选阈值 红40/蓝26）
   │      ├─ 形态学闭运算(3×3)
   │      ├─ 找轮廓（findContours）
   │      ├─ 形状筛选（面积>50 / 宽高比>1.8 / 倾斜角<65°）
   │      ├─ 求中轴线（fitLine 最小二乘拟合）
   │      ├─ 求初始端点（轮廓向轴线投影，取极值）
   │      ├─ 求最终端点（调用 refineEndpoint，用灰度梯度亚像素精修端点）   
   │      └─ 颜色分类（对轮廓边界像素的 R、B 分别求和并比较）
   │      → 返回 vector<Light>
   ├─ ③ 颜色过滤（只保留 detect_color 的灯条）
   ├─ ④ 灯条配对：按面积降序，任意两灯条满足 平行差<20° + 长度比>0.75
   └─ ⑤ 输出 ArmorResult：4 角点（左上/右上/右下/左下）+ size + 左右灯条
```

> 坐标系注意：`findLights` 算出的端点/角点是 **ROI 局部坐标**，`extractArmor` 最后配对时 `+offset`（ROI 原点在原图的位置）转回原图坐标。

## 3. 逐个函数的参数说明

### `refineEndpoint(const cv::Mat& gray, cv::Point2f endpoint, cv::Point2f dir_out)`

| 参数 | 类型 | 含义 |
|---|---|---|
| `gray` | `const cv::Mat&` | 灰度图（CV_8UC1，BGR 转出来的），用于沿灯条轴取灰度剖面、算梯度 |
| `endpoint` | `cv::Point2f` | 待精修的二值化初始端点（ROI 局部坐标），是投影极值算出来的 |
| `dir_out` | `cv::Point2f` | 指向灯条外侧的方向向量（`top - mid` 或 `bottom - mid`）；函数内部会归一化成单位向量，用来确定「沿哪条线采样灰度」 |

> 返回：精修后的亚像素端点 `cv::Point2f`。

### `findLights(const cv::Mat& roi, int detect_color)`

| 参数 | 类型 | 含义 |
|---|---|---|
| `roi` | `const cv::Mat&` | 裁剪出来的 ROI（BGR 三通道，extractArmor 外扩 10% 后的那块），灯条提取就在这块小区域里做 |
| `detect_color` | `int` | 检测的颜色：`1`=红、`2`=蓝。决定用哪个灰度阈值（红40/蓝26）和颜色分类判定 |

> 返回：`std::vector<Light>`（该 ROI 里找到的所有灯条）。

### `extractArmor(const cv::Mat& bgr, const cv::Rect2f& box, int detect_color)`

| 参数 | 类型 | 含义 |
|---|---|---|
| `bgr` | `const cv::Mat&` | 整帧原图（BGR 三通道），用于裁 ROI 和后面颜色/透视取数字 |
| `box` | `const cv::Rect2f&` | YOLO 检测框（原图坐标，含 x/y/width/height），据此外扩 + 裁 ROI |
| `detect_color` | `int` | 检测的颜色：`1`=红、`2`=蓝，传给 findLights 和颜色过滤 |

> 返回：`ArmorResult`（`valid` + `corners[4]` + `size` + `left_light`/`right_light`）。

## 4. 总结

- `detect_color` 贯穿三层（1红/2蓝），从 `extractArmor` 一路传到 `findLights`；
- `bgr`/`box` 只在 `extractArmor` 层用（裁出 `roi`），`findLights` 只关心裁出来的 `roi`；
- `gray`/`endpoint`/`dir_out` 是 `refineEndpoint` 专属的精修参数，在 `findLights` 内部调用时才传。
