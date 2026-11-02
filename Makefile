# 常用命令的一行封装；不改变 CLI 契约，只是少敲几个字。
PYTHON ?= python

.PHONY: help version test run summary findings grid visualize

help:
	@echo "make version    打印包版本"
	@echo "make test       运行完整测试套件"
	@echo "make run        跑完整流水线并落盘（约 100 秒）"
	@echo "make summary    只读：版本 / 计数 / 结论概览"
	@echo "make findings   只读：逐条结论与证据"
	@echo "make grid       只读：扇出网格与完整性"
	@echo "make visualize  写 out/report.html"

version:
	$(PYTHON) main.py --version

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) main.py run

summary:
	$(PYTHON) main.py summary

findings:
	$(PYTHON) main.py findings

grid:
	$(PYTHON) main.py grid

visualize:
	$(PYTHON) main.py visualize
