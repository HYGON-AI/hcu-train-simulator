# hcu-train-simulator 测试用例

## ST测试用例

在仓库根目录下，执行：

```shell
# 测试dense模型训练系统模拟
python -m unittest tests.st.test_dense_simulation -v

# 测试moe模型训练系统模拟
python -m unittest tests.st.test_moe_simulation -v
```