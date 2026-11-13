# 削皮第二轮真机部署计划（peel-v2，2026-11-13）

七个模型（pi0.5 LoRA，fitz0401/fr3_peel2_mix = 11-09 采集的 25 对、9 个柠檬摆放、新削皮器/GelSight 安装，seed 0）。
分支 `deploy/peel-v2` = `deploy/peel-v1` + 三处改动：服务端自动找包内的 norm stats（`fr3_peel2_mix`）、
夹爪输入按包里写的值固定（见第 1 节）、新安装的力模型。启动方式、端口（8005）、runner 用法和第一轮完全一样。

## 1. 和第一轮的区别

| 项目 | peel-v1（第一轮） | peel-v2（这一轮） |
|---|---|---|
| 数据 | fr3_peel_mix，21 对，同一摆放 | fr3_peel2_mix，25 对，9 个摆放，留出一个内部摆放（第 12/20/21 对） |
| 力模型 | 旧削皮器负载 | **新安装的负载拟合**（`fr3_train/peel2/force_model.json`，对训练力表 1912 帧逐帧核验，满 bin 的帧差值 0） |
| 夹爪输入 | 训练常数 0.12334802 | **固定为 0.28634360**（见下） |
| 模型 | 6 个，9999 步 | 7 个，9000–9999 步（见第 2 节） |

为什么夹爪还是固定：新数据里夹爪读数在每个回合内恒定，但随每次夹持取 0.2863 / 0.2907 / 0.2952 三档之一（传感器量化步长）。
runner 用 `--no-gripper`（不动夹爪，否则会松开削皮器），此时上报的读数是 0，对模型是从没见过的值。
所以服务端照旧固定一个训练值：0.2863 是最常见的一档（25 个示教回合里 14 个，含留出的第 12、21 对）。
离线测得：三档之间切换，模型动作平均变 5–8 mrad/s（held-out 误差的 15–30%），说明模型把这个读数当成了回合标识。
固定成训练值是分布内输入；真机上如果夹持后读数正好是 0.2863 最好，不是也没关系，服务端不用它。

## 2. 权重

Leonardo：`/leonardo_scratch/large/userexternal/ydu00001/openpi/deploy/pi05_peel2_<模型>_s0_<步数>`
（每个约 6.4 GB：params、norm stats、`force_model.json`、`golden.npz`、`BUNDLE.json`）。

| 模型 | 包名 | 训练方式 |
|---|---|---|
| decl_corr（ours） | pi05_peel2_decl_corr_s0_9500 | 学到的声明标签 + 回放帧的纠正动作标签 |
| moddrop | pi05_peel2_moddrop_s0_9999 | 拼接帧 + 随机去掉相机/力/触觉（p=0.3） |
| mix | pi05_peel2_mix_s0_9999 | 拼接 + 示教 + 回放，按帧声明缺失（真值标签） |
| splice | pi05_peel2_splice_s0_9500 | 只用拼接帧 |
| decl（ours） | pi05_peel2_decl_s0_9999 | 同 mix，标签换成学到的声明分类器 |
| twin（ours） | pi05_peel2_twin_s0_9999 | 拼接帧 + 与原始示教帧的一致性损失 |
| ronly2 | pi05_peel2_ronly2_s0_9000 | 只用回放（下界） |

splice、decl_corr 停在 9500 步，ronly2 停在 9000 步：最后一段训练还在排队。验证误差 4000 步以后就平了，差几百步不影响真机测试。

传输（和第一轮同一方法；自己电脑经 tar 管道中转不行）：

```bash
# 自己电脑上：证书有效期约 12 小时，过期就重新登录
step ssh login '<CINECA 邮箱>' --provisioner cineca-hpc
ssh -A <用户名>@<GPU 机>
```

```bash
# GPU 机上
ssh-add -l    # 必须能看到 CINECA 证书；看不到说明 -A 转发没生效，回到上一步
ssh -o ConnectTimeout=15 ydu00001@data.leonardo.cineca.it ls /leonardo_scratch/large/userexternal/ydu00001/openpi/deploy | grep peel2
              # 几秒内应列出 7 个 pi05_peel2_* 目录；超时就把主机名换成 login.leonardo.cineca.it
for b in decl_corr_s0_9500 moddrop_s0_9999 mix_s0_9999 splice_s0_9500 decl_s0_9999 twin_s0_9999 ronly2_s0_9000; do
  rsync -a --info=progress2 \
    ydu00001@data.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ydu00001/openpi/deploy/pi05_peel2_${b} \
    /mnt/data/yurui/models/
done
du -sb --apparent-size /mnt/data/yurui/models/pi05_peel2_*
```

`--info=progress2` 不能省：不加它 rsync 拷贝期间没有任何输出，看起来像卡住。断了重跑同一条命令即可续传。
时间紧可以先拷前四个（decl_corr、moddrop、mix、splice）。大小应为（字节）：

- pi05_peel2_decl_corr_s0_9500 6442130355 (18 个文件)
- pi05_peel2_moddrop_s0_9999 6442135831 (19 个文件)
- pi05_peel2_mix_s0_9999 6442134918 (19 个文件)
- pi05_peel2_splice_s0_9500 6442143960 (18 个文件)
- pi05_peel2_decl_s0_9999 6442121901 (20 个文件)
- pi05_peel2_twin_s0_9999 6442146520 (17 个文件)
- pi05_peel2_ronly2_s0_9000 6442141117 (16 个文件)

文件数不同是权重分片差异，正常。七个包都在 Leonardo 上通过了自检和 golden 检查（动作差 0，力输入差 0）。

### 2b. 不经过笔记本：直接在 GPU 机上登录 CINECA 并拉取

笔记本连实验室网络时用不了 Leonardo 上的对话，可以全部在 GPU 机上做。需要在 GPU 机上装 `step` 并登录一次 CINECA。

**先在笔记本上**（还连着 Leonardo 时）抄下两项配置：

```bash
cat "$(step path)/config/defaults.json"    # 抄下 "ca-url" 和 "fingerprint" 两个值
```

**然后在 GPU 机上**：

```bash
# 1) 装 step（已装过就跳过：which step 有输出即可）
wget https://dl.smallstep.com/cli/docs-cli-install/latest/step-cli_amd64.deb && sudo dpkg -i step-cli_amd64.deb
#    没有 sudo：从 https://github.com/smallstep/cli/releases 下载 step_linux_*_amd64.tar.gz，解压后把 bin/step 放进 ~/bin

# 2) 指向 CINECA 的证书服务（两个值照抄笔记本的 defaults.json），只需做一次
step ca bootstrap --ca-url <ca-url> --fingerprint <fingerprint>

# 3) 确保有 ssh-agent（ssh-add -l 报 "Could not open a connection" 时才需要）
eval "$(ssh-agent -s)"

# 4) 登录：会打开浏览器走 CINECA 账号 + 动态码；证书约 12 小时有效
step ssh login '<CINECA 邮箱>' --provisioner cineca-hpc
ssh-add -l    # 应看到 CINECA 证书

# 5) 连接测试（第一次会问是否信任主机，输入 yes）
ssh -o ConnectTimeout=15 ydu00001@data.leonardo.cineca.it ls /leonardo_scratch/large/userexternal/ydu00001/openpi/deploy | grep peel2
```

之后的 rsync 命令和上面完全一样（在 GPU 机上执行）。GPU 机是共用机器的话，用完可以 `ssh-add -D` 把证书从 agent 里删掉。

## 3. GPU 机自检（每个包一次，不动机器人）

先在 fr3_demo 里切到这个分支：`git fetch origin && git checkout deploy/peel-v2`。

```bash
cd $HOME/openpi_wine   # OpenPI 15a9616 + fr3_train/deploy.patch，和第一轮同一个环境
CUDA_VISIBLE_DEVICES=<空闲卡> PYTHONPATH=<fr3_demo> .venv/bin/python <fr3_demo>/fr3_train/serve_peel.py \
    --checkpoint-dir /mnt/data/yurui/models/pi05_peel2_decl_corr_s0_9500 --self-test \
    --golden /mnt/data/yurui/models/pi05_peel2_decl_corr_s0_9500/golden.npz
```

期望看到：`gripper input pinned to 0.28634360432624817 (fitz0401/fr3_peel2_mix)`、各类错误请求被拒绝、
`gripper reading ignored (pinned)`、`self-test OK`、`golden check PASS`。
Leonardo A100 上七个包的动作差都是 0；换 GPU 型号 bf16 舍入约 1e-3，容差 2e-2。
**如果看到 `fitz0401/fr3_peel_mix` 或 0.12334802，说明 fr3_demo 还在 peel-v1 分支。**

启动服务（一次一个模型，换模型要重启）：

```bash
start_checkpoint.sh peel_force /mnt/data/yurui/models/pi05_peel2_decl_corr_s0_9500
```

## 4. 机器人端

前提：
- 削皮器和 GelSight 按 **11-09 采集时的新安装方式**装好（夹爪不会被打开或指令）。
- `config.toml` 的 `[gel]` / `[tactile]` 用 11-09 采集时的配置。
- 起始位姿和采集时一致（数据里起始位姿固定）。
- 柠檬用台钳夹住，和采集时一样。

先做一次不动机器人的检查，再执行：

```bash
fr3-pi05-run --checkpoint peel_force --prompt 'peel the lemon' --check
fr3-pi05-run --checkpoint peel_force --prompt 'peel the lemon' --execute --auto-stop \
    --trial-log trials/peel_v2/logs/<模型>_<摆放>_<序号>.json
```

启动时提示 "Zeroing the GelSight: keep the tool still and out of contact" 时，削皮器不要碰任何东西（这一帧是触觉参考）。
凝胶保护保持开启。`--auto-stop` 对削皮只有停滞（30 s）和超时（240 s）生效；正常结束由操作员按 Back / Ctrl-C。

## 5. 测试设计

这一轮要回答的是**换位置能不能找到柠檬并削**（论文 H4），所以按摆放分三档，每档每个模型次数相同：

| 档 | 摆放 | 意义 |
|---|---|---|
| P0 | 训练用过的一个摆放（任选一个，全程固定） | 复现 |
| P1 | 留出的那个内部摆放（第 12/20/21 对的位置） | 训练网格内、没见过的位置 |
| P2 | 训练范围外 3–5 cm 的一个位置（全程固定） | 外推 |

- 必测四个：decl_corr、moddrop、mix、splice；有时间再测 decl、twin、ronly2。
- 每个模型每档 3 次（四个模型共 36 次）。为抵消柠檬磨损，按 A B C D D C B A 往返排序，每档内部同样。
- 离线结果供参考：留出位置上接近段误差是切削段的三倍（所有模型），ModDrop 最低，decl_corr 次之；
  没有模型在相机正常时读触觉。所以真机上最可能的区别在"能不能到达柠檬"。

`trials/peel_v2/results.csv` 手动记：

| 列 | 含义 |
|---|---|
| trial | 序号 |
| model | decl_corr / moddrop / mix / splice / decl / twin / ronly2 |
| placement | P0 / P1 / P2 |
| reached | 削皮器是否到达柠檬（0/1） |
| strokes | 有接触的削皮刀数 |
| peeled | 是否削下果皮（0/1） |
| peel_len_cm | 削下果皮的大致长度 |
| end | 正常结束 / 凝胶保护停 / 安全停 / 停滞 / 超时 / 人工中止 |
| note | 其它（偏离、卡住、抖动等） |

日志（`logs/*.json`、`logs/*.npz`）会记下每次推理的力窗口、触觉读数和结束原因，接触次数之后可以从日志客观算出来。

## 6. 回传

把 `trials/peel_v2/`（results.csv、logs/*.json、logs/*.npz）提交到 `deploy/peel-v2` 并推送，不要合并到 main。
