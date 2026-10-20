# ModelRouter-RSL

人工智能设计与应用作业二，选项A：在课程的ModelRouter上添加RSL。

任务是处理订单：筛选有效地区和数量，按要求排序，计算订单数和总金额。模型不做微调，学习的是保存在提示词中的经验规则。

## 做了什么

模型回答 → Python检查 → 最多修正两次 → 总结经验 → 验证通过后保存。

- 允许模型先列计算过程，最后给出JSON。程序保留原始回答，只取末尾唯一的JSON代码块检查，不改模型给出的数值。
- 训练题答错时，反馈具体错误字段及训练标签；这属于带标签训练，不能作为测试效果。
- 验证题每题只回答一次。新经验必须让通过数增加，且不能让原来正确的题变错，才能保存，最多5条。
- 测试题不提供标准答案值，不更新经验。反馈组只得到错误类别；经验组只回答一次。
- 每次修订接着上一次回答继续，最终保留得分最高的答案。

新增代码主要在 `app/core/rsl.py` 和 `app/api/rsl.py`，`homework.py` 是运行入口。`app/` 中其余文件大多来自课程框架。模型请求仍经过ModelRouter。

## 实验结果

2026年10月20日，Runpod A40、Ollama 0.40.6、Qwen2.5 14B Instruct Q4_K_M。12题训练、8题验证、20题测试，每题12条订单。最终对象的格式、编号与顺序、数量和金额全部正确才算通过。

| 方法 | 完全正确 | 平均调用次数 | 平均调用耗时 |
|---|---:|---:|---:|
| 直接回答 | 12/20（60%） | 1.00 | 27.85秒 |
| 反馈修正 | 13/20（65%） | 1.75 | 51.41秒 |
| 带经验单次回答 | 12/20（60%） | 1.00 | 27.85秒 |

反馈修正比直接回答多答对1题。没有经验通过验证，经验组复用基线；本轮未证明跨题经验学习的收益。

原先直接输出JSON的版本只有1/8通过。改进后换用了新的测试题，并调整了提示和输出流程，因此不能将两轮百分比相减，当成RSL的单独收益。RSL效果应看上表中同一批20题的组间对比。

完整原始回答、计算过程和判断见 [实验记录](results/README.md)。13项离线测试通过。

## 运行

需要Python 3.12和可用的模型服务。在本文件夹打开PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env` 中的模型地址、模型名和本地访问口令。`OPENAI_KEY_1` 对本地Ollama可填 `ollama`，云端兼容API则填写自己的密钥。不要上传 `.env`。

本机Ollama的默认地址是 `http://127.0.0.1:11434/v1`，只有安装并启动Ollama后才可用。本次实验使用的临时云端机器已释放，仓库不包含模型权重或持续可用的模型服务。

如使用Ollama复现，先在有足够资源的机器上下载模型，设置8192上下文，再启动服务。例如在Windows新PowerShell中：

```powershell
$env:OLLAMA_CONTEXT_LENGTH="8192"
ollama serve
```

如果Ollama已运行，需先退出原服务再按上述设置启动。另一个终端运行 `ollama pull qwen2.5:14b-instruct`。

启动作业服务：

```powershell
.\.venv\Scripts\python.exe homework.py serve
```

保持服务运行，在另一个终端依次执行：

```powershell
.\.venv\Scripts\python.exe homework.py learn
.\.venv\Scripts\python.exe homework.py evaluate
```

服务地址 `http://127.0.0.1:8000`，接口说明在 `/docs`。使用单进程，评测时不要同时学习。结果保存到 `results/learn-时间戳.json` 和 `results/evaluate-时间戳.json`。

首次运行从空经验开始；再次学习会读取同模型、同数据版本的已有经验。要完全重做实验，把 `.env` 中的 `ROUTER_DB`、`RSL_OUTPUT` 分别改成新的文件和目录。更换模型配置也要使用新数据库。

不接模型也可以检查流程：

```powershell
.\.venv\Scripts\python.exe homework.py demo
.\.venv\Scripts\python.exe -m pytest -q
```

`demo` 是假模型演示，不是实测。真实学习最坏约240次模型调用，测试最多80次；通常会提前停止，付费接口会产生费用。

## 局限与参考

数据为小规模合成订单，不能代表通用能力。验证集反复用于筛选经验，有过拟合风险。计算过程只供模型复核，程序不执行它；可能出现过程和最终答案不一致。评分只针对最终答案。

- 基础框架：[课程ModelRouterCourse](https://github.com/NewYearEve-Godsheep/ModelRouterCourse)。
- 课程提供的递归自我改进资料。
