# yiwise-raft-solution

## 本地评测方法
- data/test 文件夹中是本地测试集（不全），缺少的标签用 `null` 表示，在评测时会跳过
- 待评测的结果文件放在一个文件夹中，每个文件命名为与任务同名的 csv 文件，文件格式参照提交结果的格式
- src/data/eval_results.py 是本地评测脚本，命令行参数或者终端输入指定待评测结果文件夹的地址