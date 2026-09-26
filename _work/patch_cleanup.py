"""One-off: add the interface-completeness test and fix the README parameter rows."""
import ast
import io

# ---------------------------------------------------------------- 1) test additions
p = r"D:\项目\空洞填补\test_viewfill.py"
s = io.open(p, encoding="utf-8").read()
anchor = "    n_fail = sum(1 for _, ok, _ in results if not ok)"
assert anchor in s, "test anchor missing"
add = '''    # ------------------------------- 8) interface completeness and default consistency
    import dataclasses
    import inspect
    from viewfill import FillConfig
    sig = inspect.signature(fill_holes)
    fields = {f.name for f in dataclasses.fields(FillConfig)}
    internal = {"repair_warp", "repair_threshold_pct", "dev_threshold_gray", "ablate"}
    missing = sorted(fields - set(sig.parameters) - internal)
    check("fill_holes exposes every applicable FillConfig knob",
          not missing,
          (f"missing {missing}" if missing else
           f"{len(sig.parameters)} parameters, {len(fields)} config fields, "
           f"{len(internal)} intentionally internal"))
    kw = {k: v.default for k, v in sig.parameters.items()
          if v.default is not inspect.Parameter.empty and k != "return_info"}
    check("passing every documented default explicitly changes nothing",
          np.array_equal(fill_holes(rgb, inv, **kw), fixed),
          f"{len(kw)} keyword defaults round-tripped bit-exactly")
    check("return_info exposes the documented result keys",
          {"image", "filled", "warped", "hole_mask", "remaining", "crack", "oofa",
           "disocc", "filled_depth", "warped_depth", "stats"} <= set(info),
          f"{len(info)} keys")

'''
s = s.replace(anchor, add + anchor, 1)
io.open(p, "w", encoding="utf-8").write(s)
ast.parse(s)
print("test_viewfill.py: interface checks added")

# ---------------------------------------------------------------- 2) README rows
p2 = r"D:\项目\空洞填补\README.md"
lines = io.open(p2, encoding="utf-8").read().split("\n")
out, seen_epi, seen_pen, inserted = [], False, False, False
for ln in lines:
    if ln.startswith("| `epipolar` |"):
        out.append("| `epipolar` | `None` | 硬性限制匹配器相对**几何反投影行**的偏离行数。**实测它并不能改善结果**"
                   "（全图/带内/横杆行 PSNR：`None` 28.40/24.66/24.64、`2` 28.33/24.30/24.29、"
                   "`0` 28.12/23.40/**19.92**）：disocclusion 的正确背景在参考图同一行上本就被遮挡，"
                   "同行候选池装不下它。仅作逃生口，一般用 `struct_pen` 代替 |")
        seen_epi = True
        continue
    if ln.startswith("| `depth_dilate` |") and not inserted:
        out.append(ln)
        out.append("| `struct_pen` | `8.0` | **结构感知跨行惩罚**：允许从别的行借用源块，但按 "
                   "`struct_pen · w · dy²` 收费（`w` = 空洞邻域水平结构强度，`dy` = 行偏移）。竖直同质背景 "
                   "`w≈0`（好匹配不受影响），栏杆处 `w` 大（位移被罚掉）。实测（全图/带内/横杆行 PSNR，行偏移 p90/max）："
                   "`0` 28.40/24.66/24.64, 22.5/49；`0.5` 28.26/24.00/25.08, 4.0/21；"
                   "**`8`（默认，横杆对齐最好）28.41/24.74/24.42, 1.0/8** |")
        inserted = True
        continue
    if ln.startswith("| `struct_pen` |"):
        seen_pen = True
    out.append(ln)
assert seen_epi, "epipolar row not found"
assert inserted or seen_pen, "depth_dilate row not found"
io.open(p2, "w", encoding="utf-8").write("\n".join(out))
print("README.md: epipolar row corrected, struct_pen row added")
