# Wipe V2 双臂训练

此仓库包含双臂 RDP（AT → LDP）、DP 和 UViP（Teacher → utility labels → Student）的训练配置和 Slurm 脚本。克隆后按手册初始化 RDP 子模块、配置服务器路径并启动训练。[训练交接手册](configs/real_world/wipe_v2/README.md)

- HKU 的 Zarr 目录为 `/public/home/xuxinlei/RDP_wipe_11.1/V2/wipe_dish_v2_pca_v1`。[Wipe V2 数据](https://www.modelscope.cn/datasets/Xuxinlei/wipe_11.1/tree/master/V2)
- HKU 的 GelSight 原图目录为 `/public/home/xuxinlei/RDP_wipe_11.1/V2/gelsight_raw`。UViP 使用这两路原图和 Zarr 的三路场景 RGB。[GelSight 原图归档](https://www.modelscope.cn/datasets/Xuxinlei/wipe_11.1/tree/master/V2/gelsight_raw)
