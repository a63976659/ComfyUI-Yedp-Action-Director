import os
import torch
import numpy as np
import folder_paths
from server import PromptServer
from aiohttp import web
from PIL import Image
import base64
import io
import json
import hashlib
import uuid

# --- 文件夹配置 (CONFIGURATION) ---
if "yedp_anims" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yedp_anims"] = ([os.path.join(folder_paths.get_input_directory(), "yedp_anims")], {".glb", ".fbx", ".bvh"})

if "yedp_envs" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yedp_envs"] = ([os.path.join(folder_paths.get_input_directory(), "yedp_envs")], {".glb", ".gltf", ".fbx", ".obj"})

if "yedp_cams" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yedp_cams"] = ([os.path.join(folder_paths.get_input_directory(), "yedp_cams")], {".glb", ".fbx"})

# 全局缓存，用于处理海量数据负载
YEDP_PAYLOAD_CACHE = {}

class YedpActionDirector:
    """
    ComfyUI-Yedp-Action-Director (V9.20 版本 - 支持环境与 Alpha 遮罩)
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        # 节点组件名称已完全中文化
        return {
            "required": {
                "宽度": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "高度": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "总帧数": ("INT", {"default": 48, "min": 1, "max": 3000}),
                "帧率": ("INT", {"default": 24, "min": 1, "max": 60}),
                "客户端数据": ("STRING", {"default": "", "multiline": False}), 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("姿态图(POSE)", "深度图(DEPTH)", "边缘图(CANNY)", "法线图(NORMAL)", "着色图(SHADED)", "遮罩图(ALPHA)")
    FUNCTION = "render"
    CATEGORY = "Yedp/动作捕捉"
    
    DESCRIPTION = "在 ComfyUI 中控制多个 3D 角色、环境道具 (GLTF/FBX) 以及运镜关键帧。"

    @classmethod
    def IS_CHANGED(cls, 宽度, 高度, 总帧数, 帧率, 客户端数据=None, unique_id=None):
        if 客户端数据:
            return hashlib.md5(客户端数据.encode()).hexdigest()
        return float("NaN")

    def decode_batch(self, b64_list, width, height, debug_name="batch"):
        tensor_list = []
        
        for i, b64_str in enumerate(b64_list):
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            
            try:
                image_data = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                if image.size != (width, height):
                    # 兼容不同版本的 Pillow
                    resample_mode = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                    image = image.resize((width, height), resample_mode)
                    
                img_np = np.array(image).astype(np.float32) / 255.0
                tensor_list.append(torch.from_numpy(img_np))
            except Exception as e:
                print(f"[Yedp] 第 {i} 帧发生错误: {e}")
                tensor_list.append(torch.zeros((height, width, 3)))

        if not tensor_list:
            return torch.zeros((1, height, width, 3))

        return torch.stack(tensor_list)

    # 这里的传参必须与 INPUT_TYPES 中的中文键名严格对应
    def render(self, 宽度, 高度, 总帧数, 帧率, 客户端数据=None, unique_id=None):
        # 映射回内部变量
        width = 宽度
        height = 高度
        frame_count = 总帧数
        fps = 帧率
        client_data = 客户端数据

        # 1. 检查数据是否存在
        if not client_data or len(client_data) < 10:
            print("[Yedp] 错误: 未从前端 UI 接收到图像数据。")
            red_frame = torch.zeros((1, height, width, 3))
            red_frame[:,:,:,0] = 1.0 
            return (red_frame, red_frame, red_frame, red_frame, red_frame, red_frame)

        # 2. 检查是否为内存缓存 ID 而不是原始 JSON
        global YEDP_PAYLOAD_CACHE
        if client_data.startswith("yedp_payload_"):
            if client_data in YEDP_PAYLOAD_CACHE:
                client_data = YEDP_PAYLOAD_CACHE[client_data]
            else:
                print(f"[Yedp] 错误: 在内存缓存中未找到 Payload ID {client_data}！请重新点击节点上的“烘焙(BAKE)”按钮。")
                red_frame = torch.zeros((1, height, width, 3))
                red_frame[:,:,:,0] = 1.0 
                return (red_frame, red_frame, red_frame, red_frame, red_frame, red_frame)

        # 3. 解析 JSON
        try:
            data = json.loads(client_data)
        except json.JSONDecodeError as e:
            print(f"[Yedp] JSON 解析错误。")
            raise ValueError("解析来自客户端的 JSON 数据失败。")

        # 4. 解码通道批次
        pose_batch = self.decode_batch(data.get("pose", []), width, height, "pose")
        depth_batch = self.decode_batch(data.get("depth", []), width, height, "depth")
        canny_batch = self.decode_batch(data.get("canny", []), width, height, "canny")
        normal_batch = self.decode_batch(data.get("normal", []), width, height, "normal")
        shaded_batch = self.decode_batch(data.get("shaded", []), width, height, "shaded")
        alpha_batch = self.decode_batch(data.get("alpha", []), width, height, "alpha")
        
        print(f"[Yedp] 成功渲染了 {len(pose_batch)} 帧 (共 6 个通道批次)。")
        return (pose_batch, depth_batch, canny_batch, normal_batch, shaded_batch, alpha_batch)

# --- API 路由 (API ROUTES) ---
@PromptServer.instance.routes.get("/yedp/get_animations")
async def get_animations(request):
    files = folder_paths.get_filename_list("yedp_anims")
    if not files:
        files = []
    return web.json_response({"files": files})

@PromptServer.instance.routes.get("/yedp/get_envs")
async def get_envs(request):
    files = folder_paths.get_filename_list("yedp_envs")
    if not files:
        files = []
    return web.json_response({"files": files})

@PromptServer.instance.routes.get("/yedp/get_cams")
async def get_cams(request):
    files = folder_paths.get_filename_list("yedp_cams")
    if not files:
        files = []
    return web.json_response({"files": files})

@PromptServer.instance.routes.post("/yedp/upload_payload")
async def upload_payload(request):
    """
    将海量的 base64 JSON 数据存储在 Python 后端内存中，
    以防止 ComfyUI 导致浏览器的 localStorage 崩溃。
    """
    raw_text = await request.text()
    payload_id = f"yedp_payload_{uuid.uuid4().hex}"
    
    global YEDP_PAYLOAD_CACHE
    YEDP_PAYLOAD_CACHE[payload_id] = raw_text
    
    if len(YEDP_PAYLOAD_CACHE) > 3:
        oldest_key = list(YEDP_PAYLOAD_CACHE.keys())[0]
        del YEDP_PAYLOAD_CACHE[oldest_key]
        
    return web.json_response({"payload_id": payload_id})