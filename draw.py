import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（避免坐标轴标签乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 从训练日志提取的核心数据 ----------------------
# 训练轮次（0-4epoch）
epochs = np.arange(0, 5)
# 各epoch训练集平均loss（取每个epoch最后一个loss值）
train_loss = [0.7618, 0.4953, 0.3546, 0.2707, 0.2110]
# 各epoch训练集平均acc（取每个epoch最后一个acc值，转成百分比）
train_acc = [67.67, 79.85, 85.84, 89.56, 92.29]
# 各epoch验证集acc（val_acc，转成百分比）
val_acc = [81.61, 83.13, 83.13, 83.39, 84.11]

# ---------------------- 绘制双轴曲线（loss + 训练/验证acc） ----------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左坐标轴：训练loss曲线（蓝色）
ax1.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', markersize=6, label='训练损失')
ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax1.set_ylabel('损失值 (Loss)', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticks(epochs)  # 坐标轴仅显示0-4epoch
ax1.grid(True, alpha=0.3)

# 右坐标轴：训练acc + 验证acc曲线（橙色/绿色）
ax2 = ax1.twinx()
ax2.plot(epochs, train_acc, 'r-', linewidth=2, marker='s', markersize=6, label='训练准确率')
ax2.plot(epochs, val_acc, 'g-', linewidth=2, marker='^', markersize=6, label='验证准确率')
ax2.set_ylabel('准确率 (%)', fontsize=12, color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_ylim(60, 95)  # 准确率坐标轴范围60%-95%，突出变化

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

# 标题与布局
plt.title('BERT_SPC模型训练损失与准确率曲线（Restaurant数据集）', fontsize=14, pad=20)
plt.tight_layout()  # 适配布局，避免标签重叠
plt.savefig('absa_bert_spc_train_curve.png', dpi=300, bbox_inches='tight')  # 保存图片（300dpi高清）
plt.show()