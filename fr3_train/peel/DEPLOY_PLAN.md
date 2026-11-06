# 削皮模型真机部署计划（peel-v1，2026-11-06）

六个削皮模型（pi0.5 LoRA，fitz0401/fr3_peel_mix，10k 步，seed 0，终点检查点 9999）上真机。
分支 `deploy/peel-v1` = `feature/gel-safe-control`（GelSight 采集、凝胶保护、回放）合入 `deploy/bc2f-v5`
（原始 1 kHz 力流、试次日志、自动停止），再加削皮配置 `peel_force`（端口 8005）。319 个单元测试通过。

## 1. 和 bc2f 部署的区别

| 项目 | bc2f_force | peel_force |
|---|---|---|
| 服务端 | `fr3_train/serve_bc2f.py` | `fr3_train/serve_peel.py` |
| 端口 | 8004 | 8005 |
| 力模型 | 夹爪负载 | 削皮器负载（`force_model.json`，已对训练力表逐帧核验，差值 0） |
| 触觉 | 无 | `observation/tactile` = GelSight 标记 (x 位移, y 位移 px, 扭转 rad, 扩张)，来自 `GelMonitor` |
| 夹爪 | 正常开合 | **不动**：`--no-gripper` 自动生效；服务端把夹爪输入固定为训练常数 0.12334802 |
| 提示词 | pour lillet into the blue cup | **peel the lemon** |

为什么固定夹爪输入：训练数据里夹爪读数恒为 0.12334802，归一化分母只有 1e-6，真实读数偏 1e-6 就会让模型看到从没见过的 token。
削皮器全程夹着，这个输入本来就不带信息，固定成训练常数正好是分布内输入。收到的读数会回传进日志。

触觉读数无效（标记跟丢、相机断流）时，runner **直接停止本次运行**，不发 0：训练数据里标记从未跟丢，发 0 等于谎报"无接触"。

## 2. 权重

Leonardo：`/leonardo_scratch/large/userexternal/ydu00001/openpi/deploy/pi05_peel_{splice,mix,moddrop,informed,konly2,ronly2}_s0_9999`
（每个约 6 GB：params、norm stats、`force_model.json`、`golden.npz`、`BUNDLE.json`）。

和 bc2f 一样用"方法一"（2026-10-20 验证可行；经自己电脑 tar 管道中转的方法不行）：

```bash
# 自己电脑上：证书有效期约 12 小时，过期就重新登录
step ssh login '<CINECA 邮箱>' --provisioner cineca-hpc
ssh -A <用户名>@<GPU 机>
```

```bash
# GPU 机上
ssh-add -l    # 必须能看到 CINECA 证书；看不到说明 -A 转发没生效，回到上一步
ssh -o ConnectTimeout=15 ydu00001@data.leonardo.cineca.it ls /leonardo_scratch/large/userexternal/ydu00001/openpi/deploy | grep peel
              # 几秒内应列出 6 个 pi05_peel_* 目录；超时就把主机名换成 login.leonardo.cineca.it
for arm in splice ronly2 informed moddrop mix konly2; do
  rsync -a --info=progress2 \
    ydu00001@data.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ydu00001/openpi/deploy/pi05_peel_${arm}_s0_9999 \
    /mnt/data/yurui/models/
done
du -sb --apparent-size /mnt/data/yurui/models/pi05_peel_*_s0_9999
```

`--info=progress2` 不能省：每个包约 6.4 GB，不加它 rsync 在拷贝期间没有任何输出，看起来像卡住。断了重跑同一条命令即可续传。
大小应为（字节）：informed 6441566058、konly2 6441575888、mix 6441567335、moddrop 6441560326、ronly2 6441567362、
splice 6441566068（文件数 18–20 个不等，是权重分片差异，正常）。

## 3. GPU 机自检（每个包一次，不动机器人）

```bash
cd $HOME/openpi_wine   # OpenPI 15a9616 + fr3_train/deploy.patch，和 bc2f 同一个环境
CUDA_VISIBLE_DEVICES=<空闲卡> PYTHONPATH=<fr3_demo> .venv/bin/python <fr3_demo>/fr3_train/serve_peel.py \
    --checkpoint-dir /mnt/data/yurui/models/pi05_peel_splice_s0_9999 --self-test \
    --golden /mnt/data/yurui/models/pi05_peel_splice_s0_9999/golden.npz
```

期望看到：`gripper input pinned to the training constant 0.12334802...`、各类错误请求被拒绝、
`gripper reading ignored (pinned)`、`self-test OK`、`golden check PASS`。A100 上动作差约 0；换 GPU 型号 bf16 舍入约 1e-3，容差 2e-2。

启动服务（一次一个模型，换模型要重启）：

```bash
start_checkpoint.sh peel_force /mnt/data/yurui/models/pi05_peel_splice_s0_9999
```

## 4. 机器人端

前提：
- 削皮器按采集时的方式夹好（夹爪不会被打开或指令）。
- GelSight 接好，`config.toml` 的 `[gel]` / `[tactile]` 配好（和采集时一致，特别是 `rotate_180`）。
- 柠檬摆放尽量和采集时一致（数据里是同一个已削过的硬柠檬）。

先做一次不动机器人的检查，再执行：

```bash
fr3-pi05-run --checkpoint peel_force --prompt 'peel the lemon' --check
fr3-pi05-run --checkpoint peel_force --prompt 'peel the lemon' --execute --auto-stop \
    --trial-log trials/peel_v1/logs/<模型>_<序号>.json
```

启动时会提示 "Zeroing the GelSight: keep the tool still and out of contact"，此时削皮器不要碰任何东西：这一帧就是触觉的参考（训练里是回合第一帧）。
凝胶保护保持开启（到 1 级就停）：人推削皮时形变峰值只到 1 级的 54%，正常削皮不会触发。
`--auto-stop` 对削皮只有停滞（30 s 不动）和超时（240 s）两条生效；正常结束由操作员按 Back / Ctrl-C。

## 5. 测试顺序与记录

建议顺序（按离线结果）：splice（接近段和闭环回放最好）→ ronly2（切削段最好）→ informed → moddrop → mix → konly2（下界，可最后或略过）。
每个模型 3–5 次，同一摆放；换模型要重启服务端。为抵消柠檬磨损，可按 A B C … C B A 往返排序。

`fr3-pi05-trials record` 是为倒液写的（会问倒出克数），削皮不要用。请在 `trials/peel_v1/results.csv` 手动记：

| 列 | 含义 |
|---|---|
| trial | 序号 |
| model | splice / ronly2 / informed / moddrop / mix / konly2 |
| reached | 削皮器是否到达柠檬（0/1） |
| strokes | 有接触的削皮刀数 |
| peeled | 是否削下果皮（0/1） |
| peel_len_cm | 削下果皮的大致长度 |
| end | 正常结束 / 凝胶保护停 / 安全停 / 停滞 / 超时 / 人工中止 |
| note | 其它（偏离、卡住、抖动等） |

日志（`logs/*.json` 和 `logs/*.npz`；`*.raw.npz` 已被 .gitignore 忽略）会自动记下每次推理看到的力窗口、触觉读数和结束原因。
接触次数和接触时长之后可以从日志里客观算出来，和手记的对照。

## 6. 回传

把 `trials/peel_v1/`（results.csv、logs/*.json、logs/*.npz）提交到 `deploy/peel-v1` 并推送，不要合并到 main。

## 7. 已知风险（离线评估，held-out 3 对）

- 误差主要在接近段和退回段（18–32 mrad），切削段各模型相近（7.8–9.5 mrad）。
- 离线没有一个模型在切削时读 GelSight（去掉触觉误差变化在 ±1% 以内）：不要期待真机上有明显的触觉驱动行为。
- 训练数据只有 18 对；训练回合误差约 4.5 mrad，held-out 约 20 mrad，泛化有限。
- 数据里的柠檬是同一个已削过的硬柠檬；换新柠檬时果皮厚度不同，接触力和触觉会和训练不一样。
