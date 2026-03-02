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

# --- CONFIGURATION ---
if "yedp_anims" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yedp_anims"] = ([os.path.join(folder_paths.get_input_directory(), "yedp_anims")], {".glb", ".fbx", ".bvh"})

# Global Cache for massive payloads
YEDP_PAYLOAD_CACHE = {}

class YedpActionDirector:
    """
    ComfyUI-Yedp-Action-Director (V9.15 Edition)
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "frame_count": ("INT", {"default": 48, "min": 1, "max": 3000}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "client_data": ("STRING", {"default": "", "multiline": False}), 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("POSE_BATCH", "DEPTH_BATCH", "CANNY_BATCH", "NORMAL_BATCH", "SHADED_BATCH")
    FUNCTION = "render"
    CATEGORY = "Yedp/MoCap"
    
    DESCRIPTION = "Controls multiple 3D characters, animations, and camera keyframes in a web-based viewport."

    @classmethod
    def IS_CHANGED(cls, width, height, frame_count, fps, client_data=None, unique_id=None):
        if client_data:
            return hashlib.md5(client_data.encode()).hexdigest()
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
                    image = image.resize((width, height), Image.LANCZOS)
                    
                img_np = np.array(image).astype(np.float32) / 255.0
                tensor_list.append(torch.from_numpy(img_np))
            except Exception as e:
                print(f"[Yedp] Frame {i} error: {e}")
                tensor_list.append(torch.zeros((height, width, 3)))

        if not tensor_list:
            return torch.zeros((1, height, width, 3))

        return torch.stack(tensor_list)

    def render(self, width, height, frame_count, fps, client_data=None, unique_id=None):
        # 1. Check if Data Exists
        if not client_data or len(client_data) < 10:
            print("[Yedp] ERROR: No image data received from frontend.")
            red_frame = torch.zeros((1, height, width, 3))
            red_frame[:,:,:,0] = 1.0 
            return (red_frame, red_frame, red_frame, red_frame, red_frame)

        # 2. Check if it's a Memory Cache ID instead of raw JSON
        global YEDP_PAYLOAD_CACHE
        if client_data.startswith("yedp_payload_"):
            if client_data in YEDP_PAYLOAD_CACHE:
                client_data = YEDP_PAYLOAD_CACHE[client_data]
            else:
                print(f"[Yedp] ERROR: Payload ID {client_data} not found in memory cache! Please click BAKE in the node again.")
                red_frame = torch.zeros((1, height, width, 3))
                red_frame[:,:,:,0] = 1.0 
                return (red_frame, red_frame, red_frame, red_frame, red_frame)

        # 3. Parse JSON
        try:
            data = json.loads(client_data)
        except json.JSONDecodeError as e:
            print(f"[Yedp] JSON Decode Error.")
            raise ValueError("Failed to parse JSON from client.")

        # 4. Decode Batches (Now with Shaded pass)
        pose_batch = self.decode_batch(data.get("pose", []), width, height, "pose")
        depth_batch = self.decode_batch(data.get("depth", []), width, height, "depth")
        canny_batch = self.decode_batch(data.get("canny", []), width, height, "canny")
        normal_batch = self.decode_batch(data.get("normal", []), width, height, "normal")
        shaded_batch = self.decode_batch(data.get("shaded", []), width, height, "shaded")
        
        print(f"[Yedp] Successfully rendered {len(pose_batch)} frames (5 batches).")
        return (pose_batch, depth_batch, canny_batch, normal_batch, shaded_batch)

# --- API ROUTES ---
@PromptServer.instance.routes.get("/yedp/get_animations")
async def get_animations(request):
    files = folder_paths.get_filename_list("yedp_anims")
    if not files:
        files = []
    return web.json_response({"files": files})

@PromptServer.instance.routes.post("/yedp/upload_payload")
async def upload_payload(request):
    """
    Stores the massive base64 JSON payload in python memory to prevent 
    ComfyUI from crashing the browser's localStorage.
    """
    raw_text = await request.text()
    payload_id = f"yedp_payload_{uuid.uuid4().hex}"
    
    global YEDP_PAYLOAD_CACHE
    YEDP_PAYLOAD_CACHE[payload_id] = raw_text
    
    # Keep cache clean (only retain the last 3 bakes to prevent RAM bloat)
    if len(YEDP_PAYLOAD_CACHE) > 3:
        oldest_key = list(YEDP_PAYLOAD_CACHE.keys())[0]
        del YEDP_PAYLOAD_CACHE[oldest_key]
        
    return web.json_response({"payload_id": payload_id})
