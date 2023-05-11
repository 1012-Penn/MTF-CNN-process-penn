import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from pyts.image import MarkovTransitionField
import tsia.plot
import matplotlib.pyplot as plt
# import numpy as np

# 修改1：修改数据路径
DATA = r"C:\Users\Levi\Desktop\Data_500"

# 读取 CSV 数据
tag_df = pd.read_csv(os.path.join(DATA, '01_part1.csv'))

# 转换时间戳为日期时间格式
tag_df['timestamp'] = pd.to_datetime(tag_df['times'], unit='ms')

# 修改2：将 'timestamp' 列设置为 DataFrame 的索引
tag_df = tag_df.set_index('timestamp')

# 设置 MTF 参数
n_bins = 8  # 设定 MTF 的分箱数为 8
strategy = 'quantile'  # 设定 MTF 分箱策略为分位数

# 将时间序列数据转化为 2D 数组
X = tag_df.values.reshape(1, -1)
n_samples, n_timestamps = X.shape

# 使用 MTF 进行数据转换
mtf = MarkovTransitionField(image_size=48, n_bins=n_bins, strategy=strategy)

# 修改3：修改 X 的数据类型为 float 类型
X = X.astype(float)

tag_mtf = mtf.fit_transform(X)

# 创建一个绘图对象2
fig2 = plt.figure(figsize=(5, 4))

# 添加一个子图
ax = fig2.add_subplot(111)

# 绘制 MTF 图像
_, mappable_image = tsia.plot.plot_markov_transition_field(mtf=tag_mtf[0], ax=ax, reversed_cmap=True)

# 添加一个颜色条
plt.colorbar(mappable_image)

# 显示图像
plt.show()

# 获得 MTF
tag_df = pd.read_csv(os.path.join(DATA, '01_part1.csv'))
tag_df['timestamp'] = pd.to_datetime(tag_df['times'], unit='ms')
tag_df = tag_df.set_index('timestamp')

image_size = 48
X = tag_df.values.reshape(1, -1)
mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins, strategy=strategy)
tag_mtf = mtf.fit_transform(X)

# 将 MTF 图像转换为张量
X_tensor = tf.convert_to_tensor(tag_mtf)

# 将标签转换为张量 这里我把原来的label替换成value
# y = tag_df['value'].values
y = tag_df['value'].values.reshape(-1, 1)

# y_tensor = tf.convert_to_tensor(y)1


# 将张量转换为numpy数组
# X_array = tf.keras.backend.eval(X_tensor)1
# y_array = tf.keras.backend.eval(y_tensor)
# y_array = tf.keras.backend.eval(tf.convert_to_tensor(y))
# y_array = tf.keras.backend.eval(tf.convert_to_tensor(y)).astype(float)1
# X_array = X_tensor.numpy().reshape(-1, image_size, image_size, 1)2
X_array = X_tensor.numpy().reshape(-1, image_size, image_size, 1)
y_array = y.astype(float)
# test
print(X.shape)  # 输出 X 的维度信息
print(y.shape)  # 输出 y 的维度信息


# 将numpy数组传递给sklearn函数(510 2331
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42)

# 构建神经网络模型（空白行那里省略了原本的layer）
model = tf.keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_cross_entropy', metrics=['accuracy'])

# 打印模型概述
model.summary()

# 给数据集打标签（
tag_label = tag_df['value'].values

# 将 MTF 图像和对应的标签转换为张量
tag_mtf_tensor = tf.convert_to_tensor(tag_mtf, dtype=tf.float32)
tag_label_tensor = tf.convert_to_tensor(tag_label, dtype=tf.float32)


# 拆分数据集为训练集和测试集，用反斜杠来拆分使代码更易读

train_size = int(X_array.shape[0] * 0.8)
train_dataset = tf.data.Dataset.from_tensor_slices((tag_mtf_tensor, tag_label_tensor)) \
    .shuffle(buffer_size=10000) \
    .batch(batch_size=32) \
    .take(train_size)
test_dataset = tf.data.Dataset.from_tensor_slices((tag_mtf_tensor, tag_label_tensor)) \
    .shuffle(buffer_size=10000) \
    .batch(batch_size=32) \
    .skip(train_size)


# 训练模型(num_epochs为迭代次数）
num_epochs = 10
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

# 绘制训练集和测试集准确率随时间的变化曲线
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
