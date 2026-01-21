import asyncio
import json
import os
import uuid
import sys
from PIL import Image
import io
import websocket
import urllib.request
import urllib.parse
from sprites import ComfyUIBackend, generate_expression, remove_background

# ASCII Colors for readable output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def print_status(msg, success=True):
    color = GREEN if success else RED
    print(f"{color}{msg}{RESET}")

async def run_test():
    print("----------------------------------------------------------------")
    print("  STARTING AUTOMATED COMFYUI INTEGRATION TEST")
    print("----------------------------------------------------------------")

    # 1. Create a dummy anchor image
    print("1. Creating dummy anchor image...")
    try:
        img = Image.new('RGB', (512, 512), color = (73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        anchor_bytes = img_byte_arr.getvalue()
        print_status("   [OK] Dummy image created.")
    except Exception as e:
        print_status(f"   [FAIL] Could not create image: {e}", False)
        return

    # 2. Initialize Backend
    print("\n2. Connecting to ComfyUI (127.0.0.1:8188)...")
    try:
        backend = ComfyUIBackend("127.0.0.1:8188")
        if backend.ws.connected:
             print_status("   [OK] WebSocket Connected.")
        else:
             print_status("   [FAIL] WebSocket not connected.", False)
             return
    except Exception as e:
        print_status(f"   [FAIL] Connection refused. Is ComfyUI running? Error: {e}", False)
        return

    # 3. Load Workflow
    print("\n3. Loading Workflow...")
    workflow_path = "../Z test.json" 
    # Try looking for Z test-2.json if it exists, otherwise fall back
    if os.path.exists("../Z test-2.json"):
        workflow_path = "../Z test-2.json"
    
    print(f"   Using workflow: {workflow_path}")
    
    try:
        with open(workflow_path, "r") as f:
            workflow = json.load(f)
        print_status("   [OK] Workflow loaded.")
    except FileNotFoundError:
        print_status(f"   [FAIL] Workflow file not found at {workflow_path}", False)
        return
    except json.JSONDecodeError:
        print_status("   [FAIL] Workflow file is not valid JSON.", False)
        return

    # 4. Modify Workflow for Safety (Force low denoise)
    print("\n4. Validating/Patching Workflow...")
    patched = False
    for node in workflow.values():
        if node.get("class_type") == "KSampler":
            current_denoise = node["inputs"].get("denoise", 1.0)
            if current_denoise > 0.65:
                print(f"   ! Denoise was {current_denoise}, patching to 0.6 for test.")
                node["inputs"]["denoise"] = 0.6
                patched = True
    
    if patched:
        print_status("   [OK] Workflow patched for test safety.")
    else:
        print_status("   [OK] Workflow settings look safe.")

    # 5. Run Generation
    print("\n5. Running Generation (This may take 30-60s)...")
    try:
        # We call the backend directly to generate raw bytes first
        # But generate_expression is easier as it handles the executor
        
        # We need a dummy client object that matches interface if we were using generate_expression directly
        # checking sprites.py logic... generate_expression calls `client.generate`
        
        # Let's call backend.generate directly for simplicity in verification
        result_img_data = backend.generate(workflow, anchor_bytes, "joy", dramatic_pose=True)
        
        if result_img_data:
             print_status("   [OK] Image data received from ComfyUI.")
             print(f"   Data size: {len(result_img_data)} bytes")
             
             # Verify it's a valid image
             test_img = Image.open(io.BytesIO(result_img_data))
             test_img.verify() # Verify structure
             print_status("   [OK] Image structure verified (Valid PNG generated).")
             
             # Save it to prove it
             test_output = "test_output_joy.png"
             with open(test_output, "wb") as f:
                 f.write(result_img_data)
             print(f"   Saved output to: {test_output}")
             
        else:
             print_status("   [FAIL] ComfyUI returned None (Generation failed).", False)
             return

    except Exception as e:
        print_status(f"   [FAIL] Generation crashed: {e}", False)
        import traceback
        traceback.print_exc()
        return

    print("\n----------------------------------------------------------------")
    print_status("  TEST COMPLETED SUCCESSFULLY")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(run_test())
