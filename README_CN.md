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

确认画面后按 `Ctrl+C` 关闭 RViz 相机程序。

## 4. 采集演示

```bash
fr3-collect
```

- 按手柄 `X`：开始录制。
- 再按一次 `X`：结束当前演示。
- 按 `Back`：退出。

数据保存在 `data/raw/session_日期_时间`。

## 5. 添加语言指令

同一场演示使用同一条指令：

```bash
fr3-annotate --data-dir data/raw/session_日期_时间 --all
```

每个 episode 使用不同指令：

```bash
fr3-annotate --data-dir data/raw/session_日期_时间
```

## 6. 转换并上传 LeRobot 数据集

```bash
hf auth login

fr3-convert \
  --data-dir data/raw/session_1 data/raw/session_2 \
  --repo-id USERNAME/DATASET_NAME \
  --output-root data/lerobot \
  --push-to-hub
```

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
