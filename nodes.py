import os
import torch
import numpy as np
import folder_paths
from server import PromptServer
from aiohttp import web
from PIL import Image, ImageOps
import base64
import io
import json
import hashlib
import time

# --- CONFIGURATION ---
if "yedp_anims" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yedp_anims"] = ([os.path.join(folder_paths.get_input_directory(), "yedp_anims")], {".glb", ".fbx"})

class YedpActionDirector:
    """
    ComfyUI-Yedp-Action-Director (Pipeline Fix)
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        anim_files = folder_paths.get_filename_list("yedp_anims")
        if not anim_files:
            anim_files = ["none"]

        return {
            "required": {
                "animation": (sorted(anim_files), {"tooltip": "Select a motion clip from the models/yedp_anims folder."}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "frame_count": ("INT", {"default": 24, "min": 1, "max": 3000}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                # MOVED TO REQUIRED: This allows the JS to find the widget and write data to it.
                # We use multiline=False to keep the UI cleaner, though it will be filled with text.
                "client_data": ("STRING", {"default": "", "multiline": False}), 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    # UPDATED: Added Image output for Normal Batch
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("POSE_BATCH", "DEPTH_BATCH", "CANNY_BATCH", "NORMAL_BATCH")
    FUNCTION = "render"
    CATEGORY = "Yedp/MoCap"
    
    DESCRIPTION = "Controls a 3D character rig in a web-based viewport."

    @classmethod
    def IS_CHANGED(cls, animation, width, height, frame_count, fps, client_data=None, unique_id=None):
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
                # Fallback black frame
                tensor_list.append(torch.zeros((height, width, 3)))

        if not tensor_list:
            return torch.zeros((1, height, width, 3))

        return torch.stack(tensor_list)

    def render(self, animation, width, height, frame_count, fps, client_data=None, unique_id=None):
        # 1. Check if Data Exists
        if not client_data or len(client_data) < 100:
            print("[Yedp] ERROR: No image data received from frontend.")
            # Return Red frames to indicate pipeline failure
            red_frame = torch.zeros((1, height, width, 3))
            red_frame[:,:,:,0] = 1.0 
            return (red_frame, red_frame, red_frame, red_frame)

        # 2. Parse JSON
        try:
            data = json.loads(client_data)
        except json.JSONDecodeError as e:
            print(f"[Yedp] JSON Decode Error. Data snippet: {client_data[:50]}...")
            raise ValueError("Failed to parse JSON from client.")

        # 3. Decode Batches (UPDATED with Normal)
        pose_batch = self.decode_batch(data.get("pose", []), width, height, "pose")
        depth_batch = self.decode_batch(data.get("depth", []), width, height, "depth")
        canny_batch = self.decode_batch(data.get("canny", []), width, height, "canny")
        normal_batch = self.decode_batch(data.get("normal", []), width, height, "normal")
        
        print(f"[Yedp] Successfully rendered {len(pose_batch)} frames (4 batches).")
        return (pose_batch, depth_batch, canny_batch, normal_batch)

# --- API ROUTES ---
@PromptServer.instance.routes.get("/yedp/get_animations")
async def get_animations(request):
    files = folder_paths.get_filename_list("yedp_anims")
    if not files:
        files = []
    return web.json_response({"files": files})
