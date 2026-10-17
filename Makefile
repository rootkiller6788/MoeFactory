# 常用命令的一行封装；不改变 CLI 契约，只是少敲几个字。
PYTHON ?= python

.PHONY: help test run summary findings visualize

help:
	@echo "make test       运行完整测试套件"
	@echo "make run        跑完整流水线并落盘（约 100 秒）"
	@echo "make summary    只读：版本 / 计数 / 结论概览"
	@echo "make findings   只读：逐条结论与证据"
	@echo "make visualize  写 out/report.html"

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) main.py run

summary:
	$(PYTHON) main.py summary

findings:
	$(PYTHON) main.py findings

visualize:
	$(PYTHON) main.py visualize
