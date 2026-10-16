# Blue-cup 力/传感器 dropout 模型：第一轮真机部署计划

分支：`deploy/bluecup-dropout`（基于 main `0174731`）。读者：机器人工作站和 GPU 机上的 Claude 会话，以及操作员。
未经用户同意不要合并进 main。

## 1. 目标

在真机上回答离线评估回答不了的三个问题：

1. 离线排名（动作误差）和真机成功率是否一致。
2. 倒液段相机被**声明缺席**时，v4 是否比 moddrop 更能把杯子倒好，失败时是否以"少倒后停住"为主。
3. 离线看到的"全盲也准"是不是轨迹回放（见第 6 节的"瓶内起始酒量"条件）。

三个模型（同一架构、同一输入契约，换权重即可）：

| 模型目录 | 说明 | 留出集动作误差 |
|---|---|---|
| `pi05_bluecup_flag_10500` | 不做 dropout 的基线 | 0.0471 |
| `pi05_bluecup_moddrop_14999` | 随机单通道 dropout（要打败的基线） | 0.0454 |
| `pi05_bluecup_alloc_fixed_14500` | 按阶段三分支 dropout（v4，我们的方法） | 0.0452 |

## 2. 本分支包含什么、在 Leonardo 上验证到哪一步

| 文件 | 作用 | 状态 |
|---|---|---|
| `fr3_train/serve_bluecup.py` | GPU 端独立加载器（同 serve_wine.py 模式：stock openpi `15a9616` + `fr3_train/deploy.patch`），含故障注入、`--self-test`、`--golden` | 输入变换与训练逐位一致（8 种故障条件全部相同，CPU 校验）；**未在 GPU 上加载过真实权重** |
| `fr3_pi05/remote/serve_pi05_zmq.py` | 新 loader `bluecup`，元数据声明 `requires_joint_torque`、`uses_exterior2`、`fault_keys`、`bundle_model` | 单测通过 |
| `fr3_pi05/remote/start_pi05_zmq_server.sh`、`start_checkpoint.sh`、`config.toml` | 新 profile `bluecup_force`，端口 8003 | 语法检查通过 |
| `fr3_pi05/policy.py` | `build_droid_observation(..., joint_torque=, fault=)` | 单测通过 |
| `fr3_pi05/faults.py` | 故障预设 + 在线倒液检测 `PourDetector`（与训练标注同一规则、同一正运动学，已逐位核对） | 单测通过 |
| `fr3_pi05/trials.py` | `fr3-pi05-trials`：排程、每次试验自动日志 `RolloutLogger`、称重记录、汇总表 | 单测通过 |
| `fr3_train/bluecup/reference_*.png`、`reference_state.json` | 训练数据第 0 条示教起点和倒液中段的三路画面与关节角/力矩 | 用于第 4 步核对 |

全部测试：`pytest tests/` 121 passed（Leonardo，无硬件）。**运行器（`fr3_pi05/runner.py`）尚未接线**，见第 3 步。

## 3. 步骤与分工

### 第 0 步：拉分支（机器人工作站和 GPU 机都要）

```bash
cd <fr3_demo> && git fetch origin && git checkout deploy/bluecup-dropout && pip install -e '.[pi05,dev]'
```

### 第 1 步：把权重拷到 GPU 机（操作员，需要 CINECA 登录凭证）

每个目录约 6 GB（`params/` + `assets/` 下的归一化统计 + `BUNDLE.json`）：

```bash
for m in pi05_bluecup_alloc_fixed_14500 pi05_bluecup_moddrop_14999 pi05_bluecup_flag_10500; do
  rsync -av --progress ydu00001@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ydu00001/openpi/deploy/$m /mnt/data/yurui/models/
done
```

### 第 2 步：GPU 端自检并启动（GPU 机上的 Claude）

沿用 wine 的 openpi 检出（`$HOME/openpi_wine`，`15a9616` + `deploy.patch`，`prepare_wine_openpi.sh` 已能生成）。

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> $HOME/openpi_wine/.venv/bin/python fr3_train/serve_bluecup.py \
    --checkpoint-dir /mnt/data/yurui/models/pi05_bluecup_alloc_fixed_14500 --self-test
# 自检打印 5 种故障条件各一次推理，并确认 4 种错误请求被拒绝，最后一行 "self-test OK"
fr3_pi05/remote/start_checkpoint.sh bluecup_force /mnt/data/yurui/models/pi05_bluecup_alloc_fixed_14500
```

换模型 = 停服务、换目录重启。三个模型都要各跑一次 `--self-test`。
（可选更强的核对：需要时让 Leonardo 端生成 `golden.npz` 放进每个模型目录，再加 `--golden <dir>/golden.npz`，它会用训练流程算出的参考动作逐条比对。）

### 第 3 步：运行器接线（机器人工作站上的 Claude，需要真机才能完整验证）

在 `fr3_pi05/runner.py` 中：

1. `--checkpoint` 增加 `bluecup_force`。凡是 `args.checkpoint == "wine_hybrid"` 的契约分支（`open_loop_horizon = 16`、`prefetch_actions = 0`、`control_hz` 必须 15、三帧本体历史 0/45/75），对 `bluecup_force` 同样生效；逐个检查这些分支（包括第 847 行附近的 gripper 逻辑）是否属于"契约"还是"wine 专属"。
2. `_validate_policy_contract`：对 `bluecup_force` 要求元数据 `model == "pi05_bluecup"`、`requires_joint_torque is True`、`uses_exterior2 is True`、`proprio_history_offsets == [0, 45, 75]`、`action_horizon == 16`、`tasks` 非空。启动时打印 `bundle_model` 与 `bundle_step`。
3. 观测：`build_droid_observation(..., joint_torque=state["tau_J"], fault=schedule.fields(detector))`。`tau_J` 缺失时报错退出（录制端 `recording.py` 已有同样要求）；第二外部相机必须在线。提示词必须是 `pour lillet into the blue cup`。
4. 新参数（仅 `bluecup_force` 可用）：`--fault {none,cams_absent,cams_black,force_absent,force_dead_undeclared,exterior1_absent,wrist_absent,exterior2_absent}`（默认 none）、`--fault-trigger {pour,always,never}`（默认 pour）、`--pour-deg`（默认 60）、`--trial-log <path>`。
5. 每个控制步：`detector.update(q, gripper.position)`，`log.step(detector.state(), schedule.active(detector), gripper.position)`。每次推理：`log.inference(server_timing["infer_ms"])`，并核对响应里的 `fault_applied` 与发送的一致，不一致立即停机。
6. 结束原因写入日志：任务完成由操作员按一个手柄键结束（`operator_stop`，和用户约定按键），干预用 Back（`operator_abort`），`SafetyViolation` → `safety`，步数上限 → `timeout`，异常 → `error`。`log.finish(...)` 放在 `finally`。
7. `RolloutLogger(meta=...)` 至少写入：`fault_preset`、`fault_trigger`、`pour_deg`、服务端 `bundle_model` 作为 `server_model`、`bundle_step`、提示词。
8. 在 `tests/test_pi05_runner.py` 补测试；故障注入只改模型输入，**不得**绕过任何安全检查（工作空间、关节余量、速度上限、看门狗）。

### 第 4 步：不动机器人的核对（机器人端 Claude + 操作员）

1. `fr3-pi05-check --checkpoint bluecup_force`：一次推理、不执行。
2. 三路相机实时画面与 `fr3_train/bluecup/reference_*_start.png` 对比：朝向（腕部和第二外部相机在训练数据里都转过 180°，对应 `config.toml` 的 `wrist_rotate_180 = true`、`external2_rotate_180 = true`）、视野、哪个相机进哪个槽位必须一致。
3. 机械臂在 home 附近时，实时 `tau_J` 应接近 `reference_state.json` 起点值（关节 2 约 −6.6 Nm、关节 4 约 24.0 Nm），符号和单位不对就停。
4. `PourDetector` 空跑：手动示教或回放一条示教，确认倒液段被识别（倾角超过 60° 时 `pouring` 为真），放下后复位。

### 第 5 步：试验（操作员，机器人端 Claude 协助）

先用水，确认安全后再用酒。

**第一轮（40 次）**：只测 v4 和 moddrop，条件 `none` 和 `cams_absent`（倒液段触发），每格 10 次：

```bash
fr3-pi05-trials plan --session r1 --models alloc_fixed moddrop --conditions none cams_absent --reps 10 --block-reps 5
fr3-pi05-trials next --session r1      # 打印该重启哪个模型、本次的运行命令
fr3-pi05-trials record --session r1 --trial <id> --rollout-log trials/r1/logs/<id>.json --target-g <满杯克数>
fr3-pi05-trials summary --session r1 --target-g <满杯克数>
```

称重：试验前空杯放秤去皮；复位时在秤上把瓶子加到起始量并记下；倒完静置约 3 秒读杯子，再称瓶子。秤有串口时加 `--scale-port /dev/ttyUSB0`（需要 pyserial），否则手动输入。倒出量 = 杯增重；洒出量 = 瓶减重 − 杯增重。

**第二轮（第一轮差别明显后）**：加 `flag` 和 `cams_black`（未声明黑屏），每格 10 次；再做第 6 节的"起始酒量"条件。

### 第 6 步：回传结果

把 `trials/<session>/schedule.csv`、`results.csv`、`logs/*.json` 和运行器改动一起提交到本分支并推送（不要提交视频）。Leonardo 端负责分析。

## 4. 判定标准

- **成功**：倒到满线附近（`--target-g` ± `--tolerance-g`，默认 15 g）、洒出不超过 5 g、瓶子放回、没有中止。
- **v4 算好**：倒液段相机声明缺席时成功率明显高于 moddrop，失败以 `under`（少倒后停住）为主，而不是 `spill`、`over` 或碰撞。
- **离线评估不可信**：`none` 条件下三个模型的成功率排序与离线动作误差排序相反。
- **退路只是回放**：把瓶内起始酒量改成示教时的一半，`cams_absent` 下倒出量明显偏离、而有视觉时正常。这一条用 `--notes "bottle=half"` 记录，单独开一个 session。

## 5. 已知限制

- 这批模型的力输入是原始 `tau_J`，离线分析显示其中没有可读的倒液信息，模型也基本不读它；真机上"手感接替视觉"预期不会出现，这一轮测的是缺席声明和联合缺席训练的效果。
- 离线评估只有一条留出示教、单种子，真机每格 10 次的置信区间也很宽（约 ±30%），第一轮只用来看方向。
- 权重在 GPU 上的加载、相机朝向、`tau_J` 单位都要靠第 2、4 步在实验室确认。
