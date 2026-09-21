"""ESPnet3 package."""
try:
    from importlib.metadata import version
    __version__ = version("espnet3")
except Exception:
    # 如果无法从包元数据获取，使用硬编码版本
    __version__ = "0.0.0"