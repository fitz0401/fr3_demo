# FR3 演示采集与 pi0.5 部署

## 1. 启动 Bamboo（实时控制机）

```bash
cd ~/bamboo
bash RunTeleopController
```

## 2. 加载环境（机器人工作站）

```bash
cd ~/fr3_demo
source .venv/bin/activate
source /opt/ros/humble/setup.bash
```

## 3. 检查相机

```bash
fr3-camera-list
fr3-camera-rviz
```

确认画面后按 `Ctrl+C` 关闭 RViz 相机程序：一台 RealSense 同时只能被一个进程
打开，`fr3-camera-rviz` 与 `fr3-collect` 不能同时运行。采集时用下面自带的预览。

## 4. 采集演示

```bash
source /opt/ros/humble/setup.bash
fr3-collect
```

`fr3-collect` 会自动打开 RViz，显示全部正在录制的相机画面，话题与
`fr3-camera-rviz` 相同（`/fr3_demo/*`）；预览不会重复打开相机，也不会因预览错误
中断录制。

- `--no-preview`：不开预览（也可在 `config.toml` 中设 `preview = false`）。
- `--preview-publish-only`：只发布话题、不启动 RViz，便于在另一台机器上查看。
- `--preview-rate-hz`：只调整预览频率。

若未 source ROS 2，采集照常运行，只会提示 `Camera preview disabled`。

- 按手柄 `X`：开始录制。
- 再按一次 `X`：结束当前演示。
- 按 `Back`：退出。

数据保存在 `data/raw/session_日期_时间`。

相机、机械臂、夹爪和动作统一对齐到 Ubuntu 单调时钟的 15 Hz 时间线；每个完成的
episode 都有 `sync_report.json`，超出同步容差会中止。图像按 DROID 的 320x180
存储，机械臂数据包含关节位置、速度和 `tau_J` 力矩。

## 5. 添加语言指令

同一场演示使用同一条指令：

```bash
fr3-annotate --data-dir data/raw/session_日期_时间 --all
```

若要用新指令覆盖所有 episode（无论是否已标注）：

```bash
fr3-annotate --data-dir data/raw/session_日期_时间 --all --replace
```

每个 episode 使用不同指令：

```bash
fr3-annotate --data-dir data/raw/session_日期_时间
```

## 6. 转换并上传 LeRobot 数据集

当前 `fr3-collect` 采集的是严格同步的 schema v3 数据，可直接转换并上传：

```bash
hf auth login

fr3-convert \
  --data-dir data/raw/session_日期_时间 \
  --repo-id USERNAME/DATASET_NAME \
  --output-root data/lerobot \
  --push-to-hub
```

`--push-to-hub` 才是上传开关。转换前程序会检查语言指令、帧数和
`sync_report.json`；检查失败的数据不会上传。

`--allow-legacy-unsynchronized` **不用于新数据，也不是上传开关**。它只允许转换本
项目加入同步采集之前产生的 schema v1 数据；这类旧数据没有 `sync_report.json`，
传感器只按循环序号配对。只有人工确认旧数据可用并愿意承担时间错位风险时，才添加
这个参数。schema v2/v3 数据不要添加。

## 7. 启动新 checkpoint（GPU 机器）

如果模型训练时使用了 L515，在 GPU 机器的 `~/fr3_demo/config.toml` 中设置：

```toml
[pi05]
use_external2 = true
```

没有使用 L515 则设置为 `false`。然后启动 checkpoint：

```bash
bash ~/fr3_demo/fr3_pi05/remote/start_checkpoint.sh \
  /mnt/data/yurui/models/NEW_CHECKPOINT
```

## 8. 执行推理（机器人工作站）

```bash
fr3-pi05-check --server-only
fr3-pi05-run --execute --prompt "xxx"
```

`fr3-pi05-run` 会询问语言指令。更换 GPU checkpoint 后，机器人工作站的命令不变。
