from .nodes import YedpActionDirector

# ComfyUI 注册节点的映射字典
NODE_CLASS_MAPPINGS = {
    "YedpActionDirector": YedpActionDirector
}

# 节点在 UI 中显示的人类可读名称 (已汉化)
NODE_DISPLAY_NAME_MAPPINGS = {
    "YedpActionDirector": "🎬 Yedp 动作导演 (Action Director)"
}

# 包含前端 JS 文件的目录，ComfyUI 会自动加载并挂载
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]