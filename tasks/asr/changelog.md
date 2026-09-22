# Changelog - TEMPLATE/asr

> 本文件记录 `TEMPLATE/asr` 的所有版本变更。
> 返回 [总纲 CHANGELOG](../CHANGELOG.md)
> 格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/)，版本号遵循 CalVer (YYMM) 规范。

---

## [v202604] - 2026-04-15

### Changed
- `encoder_type` 字段重命名为嵌套结构 `encoder.type`，与引擎侧对齐
- 默认 `batch_size` 从 16 调整为 32（A100/H100 基准测试最优值）

### Deprecated
- `hooks.on_dataset_ready(dataset)` 旧签名已废弃，新增 `config` 参数（旧签名将于 v202607 移除）

---

## [v202511] - 2025-12-01

### Removed
- 移除 `use_spec_aug` 布尔开关，SpecAugment 改为数据处理管线组件

---

## [v202509] - 2025-09-08

### Added
- 初始发布：支持 Conformer-V2
- 提供 AISHELL-1 / WenetSpeech baseline recipe