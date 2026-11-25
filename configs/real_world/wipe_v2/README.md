# Wipe V2 双臂训练

这份手册覆盖 RDP 的 AT 和 LDP 两步训练、动作空间 DP，以及 UViP 的 Teacher、utility labels 和 Student。三条流程使用同一份 Wipe V2 划分和动作标签。

## 1. 数据

- HKU 服务器已解包目录为 `/public/home/xuxinlei/RDP_wipe_11.1/V2/wipe_dish_v2_pca_v1`。该目录下应有 `replay_buffer.zarr`、`dataset_manifest.json`、`split.json` 和 `pca/`。
- 在其他机器上，从 ModelScope 下载 V2 分卷。已核对的版本是 `3cc8a53f`，合并后的 tar SHA-256 为 `54aca953d72311a8a99a5178830fc31064a5265506835c35dcfc29b5ceb8180a`。[ModelScope Wipe V2 数据](https://modelscope.cn/datasets/Xuxinlei/wipe_11.1/tree/master/V2)
- UViP 的两路 GelSight 原图位于 HKU 的 `/public/home/xuxinlei/RDP_wipe_11.1/V2/gelsight_raw`，包含 50 个 `episode_*.tar` 和 `manifest.json`。ModelScope 版本为 `cd2e2c0f`。[GelSight 原图归档](https://www.modelscope.cn/datasets/Xuxinlei/wipe_11.1/tree/master/V2/gelsight_raw)

共 50 条示范和 40,895 帧，按发布者划分使用 40 条训练、10 条验证，对应 32,766 和 8,129 个窗口。动作是下一快照采集的 14 维主臂绝对位置指令，左臂 7 维后接右臂 7 维。名义采样频率为 30 Hz。三条流程的动作窗口长 32，执行索引为 3 至 31，观测使用当前帧和前两帧处的观测。

## 2. 环境和目录

在 Linux 服务器上进入 AdaViP checkout，在整个训练期间保留相同的绝对路径。以下示例把环境、数据、日志和模型写到 checkout 外。把 `TRAIN_ROOT` 设为你在服务器上有写权限的持久目录。

```bash
git submodule update --init third_party/reactive_diffusion_policy
git -C third_party/reactive_diffusion_policy rev-parse HEAD
# 824c5e8de1fd1811106907a04b5f0186e0138c0b

export TRAIN_ROOT=/public/home/wangzihao/wipe-v2-handoff
mkdir -p "$TRAIN_ROOT"/{runs,logs,cache}
export HF_HOME="$TRAIN_ROOT/cache/huggingface"
export TORCH_HOME="$TRAIN_ROOT/cache/torch"
python3.11 -m venv "$TRAIN_ROOT/venv"
source "$TRAIN_ROOT/venv/bin/activate"
# 选用与服务器 NVIDIA 驱动兼容的 CUDA 版。下面以 CUDA 12.4 为例。
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r configs/real_world/wipe_v2/requirements.txt

export WIPE_V2_DATA=/public/home/xuxinlei/RDP_wipe_11.1/V2/wipe_dish_v2_pca_v1
test -r "$WIPE_V2_DATA/split.json" && test -d "$WIPE_V2_DATA/replay_buffer.zarr"
export WIPE_V2_REPO="$PWD"
RUN=scripts/real_world/wipe_v2/run.sh
SBATCH_SCRIPT=scripts/real_world/wipe_v2/train.sbatch
```

需要重新下载数据时，在 CPU 节点运行以下命令。分卷与合并前后的文件约需 46 GB 的存储空间。

```bash
V2="$TRAIN_ROOT/V2"
python scripts/real_world/download_modelscope_public_folder.py \
  --dataset Xuxinlei/wipe_11.1 --folder V2 --revision 3cc8a53f \
  --output "$V2" --strip-folder --evidence-dir "$TRAIN_ROOT/logs/download"
python scripts/real_world/wipe_v2/unpack.py "$V2"
export WIPE_V2_DATA="$V2/wipe_dish_v2_pca_v1"
```

## 3. 双臂 RDP 和 DP

| 阶段 | 输入 | 默认训练配置 | 产物 |
| --- | --- | --- | --- |
| RDP AT | 14 维动作、两路触觉 PCA | 601 epochs，batch 128，训练/验证 workers 8/4 | `best.ckpt`、`latest.ckpt` |
| RDP LDP | 三路 RGB、双臂状态、两路 PCA、冻结的 AT | 401 epochs，batch 64，workers 8/8 | `best.ckpt`、`latest.ckpt`、指定轮次 |
| DP | 三路 RGB、双臂状态、两路历史 PCA | 600 epochs，batch 64，workers 8/8 | `best.ckpt`、`latest.ckpt`、指定轮次 |

三个 Hydra 配置分别是 [at.yaml](at.yaml)、[ldp.yaml](ldp.yaml) 和 [dp.yaml](dp.yaml)。全部使用 seed 42。AT 的 `best.ckpt` 按训练损失选择。LDP 使用 `WIPE_V2_AT` 指定的 AT checkpoint，下面的命令选择 `best.ckpt`。

DP 和 LDP 完成第 20、40、60、100 轮时，在各自的 `checkpoints/` 下分别保留 `epoch-0020.ckpt`、`epoch-0040.ckpt`、`epoch-0060.ckpt`、`epoch-0100.ckpt`。`best.ckpt` 按验证损失更新，`latest.ckpt` 每轮更新。

如果要复现后来提交的 400 epochs DP 配方，在新的输出目录启动 DP 时追加 `training.num_epochs=400 logging.name=wipe_v2_tactile_dp64_epoch400_seed42`。

先在交互 GPU allocation 中查看 `nvidia-smi` 和正在运行的进程，报告 GPU 型号、显存、利用率和相关作业并取得确认。用新的输出目录做短跑，检查实际 batch、每轮迭代数、有限损失、GPU 利用率和 checkpoint，再根据实际占用设定正式作业的 CPU、内存与时限。

```bash
# 在已确认的交互 allocation 中运行。短跑产物保存在独立目录。
srun bash "$RUN" at "$TRAIN_ROOT/runs/at-smoke" training.num_epochs=2
srun bash "$RUN" dp "$TRAIN_ROOT/runs/dp-smoke" training.num_epochs=2
```

从 checkout 根目录提交正式作业。按账号权限给 `sbatch` 添加分区、账号和 QoS 参数，并根据预跑结果覆盖 `--mem`、`--time` 或 GPU 型号。

```bash
sbatch --output="$TRAIN_ROOT/logs/%x-%j.out" "$SBATCH_SCRIPT" at "$TRAIN_ROOT/runs/at"
sbatch --output="$TRAIN_ROOT/logs/%x-%j.out" "$SBATCH_SCRIPT" dp "$TRAIN_ROOT/runs/dp"

# 选择 AT 的 best.ckpt，LDP 的预跑和正式训练使用同一份。
export WIPE_V2_AT="$TRAIN_ROOT/runs/at/checkpoints/best.ckpt"
srun bash "$RUN" ldp "$TRAIN_ROOT/runs/ldp-smoke" training.num_epochs=2
sbatch --output="$TRAIN_ROOT/logs/%x-%j.out" "$SBATCH_SCRIPT" ldp "$TRAIN_ROOT/runs/ldp"
```

训练过程写入各自输出目录的 `logs.json.txt`。用 `sacct -j <jobid>` 确认作业状态，并以完整的验证轮数和 checkpoint 检查训练进度。LDP checkpoint 记录 AT 的路径、SHA-256 和实际 epoch，后续加载 LDP 时需保留这份 AT。

## 4. UViP Teacher → utility labels → Student

UViP 直接读取 GelSight tar。缓存阶段校验每条归档的 SHA-256，通过两路 `timestamps.csv` 的 `cleaned_frame_index` 对齐 40,895 帧。三路场景 RGB、动作、状态和划分从 Wipe V2 Zarr 读取。使用已有的 512 维语言向量可以传 `--language-embedding`，离线环境也可以通过 `--clip-model` 指定本地 CLIP 权重目录。

```bash
export WIPE_V2_GELSIGHT=/public/home/xuxinlei/RDP_wipe_11.1/V2/gelsight_raw
UVIP="$TRAIN_ROOT/runs/uvip"
bash "$RUN" uvip-init --dataset "$WIPE_V2_DATA" --gelsight "$WIPE_V2_GELSIGHT" --output "$UVIP"
bash "$RUN" cache --config "$UVIP/config.json"
bash "$RUN" prepare --config "$UVIP/config.json"
```

其他服务器可以在 CPU 节点下载同一版本，约 5.22 GB。下载后将 `WIPE_V2_GELSIGHT` 指向对应目录。UViP 图像缓存约需额外 43 GB。

```bash
export WIPE_V2_GELSIGHT="$TRAIN_ROOT/V2/gelsight_raw"
python scripts/real_world/download_modelscope_public_folder.py \
  --dataset Xuxinlei/wipe_11.1 --folder V2/gelsight_raw --revision cd2e2c0f \
  --output "$WIPE_V2_GELSIGHT" --strip-folder --evidence-dir "$TRAIN_ROOT/logs/gelsight-download"
```

如使用完整的同源 HDF5，可改传 `--hdf5 /path/to/episodes --layout cleaned`（清洗版）或 `--layout raw`（原始版）。

默认使用原 Wipe UViP 配方的 batch 256、16 workers、Teacher 500 epochs 和 Student 600 epochs。预跑与正式作业使用同一份 `config.json`。Teacher 训练结束后选取验证最优的模型，再为训练和验证集生成 utility labels。标签通过审计后启动 Student。

```bash
# 在已确认的交互 GPU allocation 中执行，检查 preflight.json 和显存记录。
srun bash "$RUN" preflight --config "$UVIP/config.json"

sbatch --output="$TRAIN_ROOT/logs/%x-%j.out" "$SBATCH_SCRIPT" teacher --config "$UVIP/config.json"
# Teacher 完成并写出 teacher_selection.json 后提交。
sbatch --output="$TRAIN_ROOT/logs/%x-%j.out" "$SBATCH_SCRIPT" labels --config "$UVIP/config.json"
# 两个 split 都写出 COMPLETE.json 后，在 CPU 节点执行。
bash "$RUN" audit --config "$UVIP/config.json"
# labels/AUDIT.json 中 passed 为 true 后提交。
sbatch --output="$TRAIN_ROOT/logs/%x-%j.out" "$SBATCH_SCRIPT" student --config "$UVIP/config.json"
```

Teacher 和 Student 中断后，可用原配置和 `--resume` 续跑。标签阶段重复提交会重用通过校验的分片。最终模型位于 `teacher/best.pt` 和 `student/best.pt`。

## 5. GitHub 交接

发布包含本 README、训练配置、Python 适配代码、命令行脚本和固定的 RDP 子模块引用的 Git 分支。数据从上面的 ModelScope 和服务器共享目录获取。模型、缓存、标签、语言向量和运行日志存放在 `TRAIN_ROOT`。
