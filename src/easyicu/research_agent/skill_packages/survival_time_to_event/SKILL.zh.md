# Survival time to event 参考脚本

使用已安装 EasyICU 的宿主估计器。输入为预处理队列（示例以唯一 stay_id
标识每行）及明确指定变量、水平、缺失处理和方法的 JSON 方案。
输出包括结果 CSV、spec.json、step_summary.json、manifest.json；关联与生存
分析另有基于结果表的效应图。估计器配置及依赖见 references/methods.md。
结果仅用于分析，不自动获得论文证据授权。

```sh
python -m easyicu.research_agent.skill_packages.survival_time_to_event --example --out ./reference_output
python -m easyicu.research_agent.skill_packages.survival_time_to_event --cohort ./prepared.parquet --spec ./reviewed.json --out ./results
```

example_data.py 仅生成合成数据；无需数据库、网络或研究运行目录。
实际分析前需准备并审阅队列和方案；脚本不代选研究人群或科学配置。
