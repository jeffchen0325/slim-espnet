"""ESPnet package."""
try:
    from importlib.metadata import version
    __version__ = version("slim-espnet")
except Exception:
    # 如果无法从包元数据获取，使用硬编码版本
    __version__ = "0.0.0"