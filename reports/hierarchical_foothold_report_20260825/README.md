# 分层落足规划总结报告

- `hierarchical_foothold_report.pdf`：34 页、16:9 Beamer 风格中文报告；正文以方法实现、训练流程和现有结果为主。
- `hierarchical_foothold_report.tex`：XeLaTeX 源文件。
- `media/`：报告引用的 7 段完整实验视频。
- `assets/`：视频序列图、典型地形图与 PDF 封面图。

编译：

```bash
xelatex -interaction=nonstopmode -halt-on-error hierarchical_foothold_report.tex
xelatex -interaction=nonstopmode -halt-on-error hierarchical_foothold_report.tex
```

PDF 中的实验图片可点击播放 `media/` 下的对应视频。部分 PDF 阅读器会禁用 `run:` 链接；此时请按最后一页视频索引直接打开文件。

报告中的完成、碰撞、物理终止和越界均为现有评估产物中的事件计数。自动 reset 后可能在同一环境内产生多个完成事件，因此报告没有把这些计数误写成 episode success rate。
