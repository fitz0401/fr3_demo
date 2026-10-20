# Blue-cup 第五版（bc2f）三模型真机对比：部署计划

分支：`deploy/bc2f-v5`（基于 `deploy/bluecup-dropout` 并合入 main `8358454`，即带 1 kHz 原始力录制的版本）。
读者：机器人工作站和 GPU 机上的 Claude 会话，以及操作员。未经用户同意不要合并进 main。

## 1. 目标

三个模型训练数据、输入、结构完全相同，只有训练时的 dropout 不同。部署时三者都拿到全部输入：

| 模型（bundle 目录） | 训练时做了什么 | 离线留出误差（mrad，全程 / 倒液 / 回程） |
|---|---|---|
| `pi05_bc2f_vanilla_s0_14999` | 不做 dropout | 49.9 / 25.2 / 60.9 |
| `pi05_bc2f_moddrop_s0_14999` | ModDrop：三路相机和力各以 0.3 的概率被随机拿掉 | 50.9 / 21.0 / 63.0 |
| `pi05_bc2f_moddrop_prop_s0_14999` | ModDrop + 本体（当前关节角和历史）以 0.5 的概率拿掉 | 51.1 / 21.5 / 66.1 |

离线结果：全程误差三者相同；倒液段差别在不同 checkpoint 间不稳定；ModDrop+本体组在回程更差，且几乎不再读本体。
**离线动作误差回答不了"倒得好不好"**，所以这一轮在真机上直接比：成功率、倒出量、洒出量、用时、
卡住或停不下来的次数，以及回程段是否出问题。

输入（三个模型相同）：三路相机（与上一轮相同的槽位）、本体（当前、3 s 前、5 s 前的关节角和夹爪）、
**力**：法兰外力（6 维，重力补偿后）在每次观测前 64 ms 内分 4 段取平均。力由 GPU 端从机器人发来的原始
~1 kHz 关节数据（`time_sec`、q、tau_J）现算，重力模型随 bundle 提供（`force_model.json`），与训练逐位一致。

## 2. 本分支包含什么、在 Leonardo 上验证到哪一步

| 文件 | 作用 | 验证状态 |
|---|---|---|
| `fr3_train/serve_bc2f.py` | GPU 端独立加载器（stock openpi `15a9616` + `fr3_train/deploy.patch`），分段提示词、力窗口、`--self-test`、`--golden` | **已在 Leonardo 上用 stock openpi + deploy.patch 跑过 `--self-test` 和 `--golden`**（见第 2.1 节） |
| `fr3_train/force_features.py` | 原始数据 → 24 维力输入（纯 numpy） | 与 pinocchio 差 < 1e-9；8 条示教 10,993 帧中，凡 16 个小段都有样本的帧与训练输入**完全相同**；仅 2 帧（各缺 1 个小段，训练时用了该帧之后的样本插值）有 ≤0.14 N 差别 |
| `fr3_train/bc2f/force_model.json` | 同 bundle 里的那份（测试用） | 同上 |
| `fr3_pi05/raw_force.py` | 运行器端后台轮询线程（自带一个 Bamboo 客户端，与录制端同样方式），给每次观测取原始窗口 | 单测（假 Bamboo）通过 |
| `fr3_pi05/runner.py` | 新 `--checkpoint bc2f_force`：启动轮询、三帧历史、观测带原始力、夹爪动作时手臂停住（与示教一致）、`--trial-log` | 单测通过；**未在真机上跑过** |
| `fr3_pi05/remote/*`、`config.toml` | 新 loader `bc2f`、profile `bc2f_force`、端口 8004 | 单测、语法检查通过 |
| `fr3_pi05/trials.py` | 新增三个模型、无目标量的成功判定（`--min-poured-g`） | 单测通过 |
| `fr3_train/bc2f/reference_*` | 本数据集第 0 条示教起点和倒液中段的三路画面、关节角、tau_J、力窗口；home 处与搬运中的力参考值 | 第 4 步核对用 |

全部测试：`python -m unittest discover -s tests` 152 个通过（Leonardo，无硬件）。

### 2.1 Leonardo 上的部署核对

`openpi_drop/scripts/bc2f_deploy_check.sbatch`：对每个 bundle，先用训练流程（训练用的力表、dropout 关闭）
在留出示教 3、10、17 的接近、搬运、倒液、回程各取一帧，固定扩散噪声，生成 `golden.npz`；再用部署加载器
（stock openpi + deploy.patch，从 golden 里的**原始**关节数据现算力）重放并比对。结果（作业 61527593，A100）：三个模型 `--self-test` 全部通过（7 种错误请求都被拒绝），`--golden` 全部 PASS，12 帧 × 3 个模型的**力输入和动作与训练流程逐位相同（差 0）**。这一步还查出并修掉了一个会让服务每次推理都崩溃的问题：stock 的 `create_trained_policy` 会用 48 维统计量去反归一化只剩 8 维的 `state`，现在加载器只反归一化动作。

## 3. 步骤与分工

### 第 0 步：拉分支（机器人工作站和 GPU 机都要）

```bash
cd <fr3_demo> && git fetch origin && git checkout deploy/bc2f-v5 && pip install -e '.[pi05,dev]'
```

如果机器人端已经在 `deploy/bluecup-dropout` 上有本地改动（例如上一轮的运行器接线），先提交，再
`git merge deploy/bc2f-v5`；本分支没有改上一轮 bluecup 的契约，只新增了 `bc2f_force`。

### 第 1 步：把权重拷到 GPU 机（操作员，需要 CINECA 登录凭证）

每个目录约 6.4 GB（`params/`、`assets/` 下的归一化统计、`force_model.json`、`golden.npz`、`BUNDLE.json`），
**整个目录**拷过去，`force_model.json` 缺了服务会拒绝启动。做法：在自己电脑上刷新 CINECA 证书，用 `ssh -A` 登录
GPU 机（把证书转发过去），再在 GPU 机上直接从 Leonardo 拉（经自己电脑用 tar 管道中转的方式试过不行）：

```bash
# 自己电脑上
step ssh login '<CINECA 邮箱>' --provisioner cineca-hpc
ssh -A <用户名>@<GPU 机>
# GPU 机上（ssh-add -l 应能看到 CINECA 证书）
for m in pi05_bc2f_vanilla_s0_14999 pi05_bc2f_moddrop_s0_14999 pi05_bc2f_moddrop_prop_s0_14999; do
  rsync -a --info=progress2 ydu00001@data.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ydu00001/openpi/deploy/$m /mnt/data/yurui/models/
done
du -sb --apparent-size /mnt/data/yurui/models/pi05_bc2f_*_s0_14999   # 6442772399 / 6442765294 / 6442754821
```

### 第 2 步：GPU 端自检并启动（GPU 机上的 Claude）

沿用 wine/bluecup 的 openpi 检出（`$HOME/openpi_wine`，`15a9616` + `deploy.patch`）。三个模型都要各跑一次：

```bash
for m in pi05_bc2f_vanilla_s0_14999 pi05_bc2f_moddrop_s0_14999 pi05_bc2f_moddrop_prop_s0_14999; do
  B=/mnt/data/yurui/models/$m
  (cd $HOME/openpi_wine && CUDA_VISIBLE_DEVICES=<空闲卡> PYTHONPATH=<fr3_demo> .venv/bin/python \
      <fr3_demo>/fr3_train/serve_bc2f.py --checkpoint-dir $B --self-test --golden $B/golden.npz)
done
# 每个模型最后两行应为 "self-test OK" 和 "golden check PASS"。A6000 与 Leonardo 的 A100 数值舍入不同，
# 动作差在 1e-3 量级属正常；力输入差应 < 1e-3（与 GPU 无关）。FAIL 就停，把日志推回分支。
fr3_pi05/remote/start_checkpoint.sh bc2f_force /mnt/data/yurui/models/pi05_bc2f_moddrop_s0_14999
```

换模型 = 停服务、换目录重启。

### 第 3 步：运行器（机器人工作站上的 Claude）

接线已经在本分支写好（`fr3_pi05/runner.py` 中搜 `bc2f_force`），请**先读一遍再用**：

1. `--checkpoint bc2f_force`：与 `wine_hybrid` 相同的契约（16 步动作块整块执行、不预取、控制频率必须
   15 Hz、三帧本体历史 0/45/75、夹爪开合时手臂停住，等夹爪动完再重新推理——本数据的示教在夹爪开合前后
   各 ≥12 帧手臂静止，已核对）。
2. 启动时另开一个 Bamboo 客户端做后台轮询（`RawForceSampler`，与录制端 `_arm_loop` 相同），要求实测频率
   ≥500 Hz（录制时 861 Hz），并打印频率和最大间隔。每次推理的观测带上
   `observation/raw_force`（最近 128 ms 的原始样本 [N, 15]）和 `observation/robot_timestamp`（作为当前关节角的那次
   状态的 `time_sec`）。GPU 端会检查二者一致（关节角差 > 0.02 rad 或窗口不全就拒绝），所以接线错了会立刻报错而不是
   悄悄喂错数据。
3. `--trial-log <path>`：写一条试验 JSON（结束原因、倒液检测、每次推理的延迟和该次力窗口均值），
   `fr3-pi05-trials record` 会读它。结束原因：手柄 Back = `operator_stop`；Ctrl+C = `interrupted`；
   安全停 = `safety`；异常 = `error`。
4. 安全检查（工作空间、关节余量、速度上限 0.35 rad/s、看门狗）一律不变。
5. Bamboo 状态里必须有 `tau_J` 和 `time_sec`（录制时已有）；没有就报错退出。

### 第 4 步：不动机器人的核对（机器人端 Claude + 操作员）

1. `fr3-pi05-run --checkpoint bc2f_force --check --prompt 'pour lillet into the blue cup'`：会归位、开夹爪、
   采 5 s 历史、做一次推理但不执行。看三行输出：原始力频率（应 ~860 Hz，≥500 Hz）、服务端模型名和步数、
   推理成功。
2. 三路相机实时画面与 `fr3_train/bc2f/reference_*_start.png` 对比：朝向、视野、哪个相机进哪个槽位必须一致。
3. **力参考值**（`reference_state.json`，法兰外力，基座坐标，N / Nm）：每条示教开头都在 home 静止约 2 秒
   （空夹爪），22 条示教这段为 F = (0.54 ± 0.28, 0.12 ± 0.11, −0.53 ± 0.33) N，
   M = (−0.14 ± 0.03, −0.43 ± 0.14, −0.12 ± 0.04) Nm；搬运瓶子时 F_z = −7.1 ± 2.0 N。
   同时看关节力矩：home 静止时 tau_J 关节 2 约 −6.4 Nm、关节 4 约 23.9 Nm（`reference_state.json` 的 start 帧）。`--check` 时加 `--trial-log /tmp/check.json`，日志里的 `force` 事件就是这次推理
   模型看到的力窗口均值。与参考值差别超过约 3 倍标准差（尤其 F_z、M_y）说明腕部负载变了（相机、线缆、夹爪换过
   或重新走线），力输入会偏出训练分布，先停下报告。
4. 如有需要，手动示教一次倒液动作（不执行策略）确认 `PourDetector` 能识别倾倒（`max_tilt_deg` > 60°）。

### 第 5 步：试验（操作员，机器人端 Claude 协助）

先用水，确认安全后再用酒。场景按示教：杯和瓶放在示教时的位置附近，瓶内起始量在示教范围内
（示教瓶重 675–974 g，中位 835 g）。

**第一轮（30 次）**：三个模型 × 每个 10 次，同一模型连做 5 次为一块（一次服务启动），块顺序随机：

```bash
fr3-pi05-trials plan --session v5r1 --models bc2f_vanilla bc2f_moddrop bc2f_moddrop_prop --conditions none --reps 10 --block-reps 5
fr3-pi05-trials next --session v5r1      # 打印该重启哪个模型、本次的运行命令
fr3-pi05-trials record --session v5r1 --trial <id> --rollout-log trials/v5r1/logs/<id>.json --min-poured-g 20 --spill-tolerance-g 10
fr3-pi05-trials summary --session v5r1
```

操作员**不要**看日志去猜是哪个模型（`next` 会打印模型名，记录时只照实填称重和结果）。

称重：试验前空杯放秤去皮；称瓶子（`bottle_before`）；倒完瓶子放回、夹爪松开后按 Back 结束；静置约 3 秒读杯子，
再称瓶子。倒出量 = 杯增重；洒出量 = 瓶减重 − 杯增重。

### 第 6 步：回传结果

把 `trials/<session>/schedule.csv`、`results.csv`、`logs/*.json` 和运行器改动一起提交到本分支并推送（不要提交视频）。
Leonardo 端负责分析。

## 4. 判定标准

- **成功**：杯中至少 20 g（示教最少约 6 g、中位 56 g、四分位 43–68 g）、洒出不超过 10 g、瓶子放回且松开、
  由操作员按 Back 正常结束。
- 主要比较：三个模型的成功率；次要：倒出量分布（与示教中位 56 g 比）、洒出量、用时、中止和安全停次数。
- 每个模型 10 次，成功率的置信区间约 ±30%，第一轮只用来看方向；差别明显再加次数。
- 特别留意：ModDrop+本体组在回程段（放回瓶子、松开、回位）是否出问题——离线它在回程误差最大。

## 5. 已知限制

- 三个模型都只用了种子 0 的第 15000 步（预先定好的规则，不是挑出来的）。
- 力输入的重力模型是这台机器、这套腕部负载（相机、线缆、夹爪）标定的；负载变了就要重新导出 `force_model.json`。
- 离线评估只有 3 条留出示教 × 2 个种子；倒液段每条只有约 48 帧，离线排序本身就不稳定。
- 示教开头都先在 home 停约 2 秒（指令速度为 0）再开始动，而 home 静止时画面、本体、力都没有"该出发了"的信号，
  所以模型可能在起点停几秒才动。起点停留超过约 10 秒请在 `--notes` 里记"start_stall"，这本身也是要比较的行为。
