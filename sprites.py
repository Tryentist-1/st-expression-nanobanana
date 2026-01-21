import os
import io
import asyncio
import time
from pathlib import Path
from typing import List, Optional

import typer
import gradio as gr
from tqdm import tqdm
from PIL import Image
from rembg import remove
from google import genai
from google.genai import types
from dotenv import load_dotenv, set_key
import uuid
import json
import urllib.request
import urllib.parse
import websocket
import random

load_dotenv()

app = typer.Typer(help="SillyTavern Sprite Generator")

EMOTIONS = [
    "neutral", "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", 
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", 
    "nervousness", "optimism", "pride", "realization", "relief", "remorse", 
    "sadness", "surprise", "blush"
]

class ComfyUIBackend:
    def __init__(self, server_address):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = websocket.WebSocket()
        self.ws.connect("ws://{}/ws?clientId={}".format(self.server_address, self.client_id))

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
             return json.loads(response.read())

    def upload_image(self, image_data, name="upload_image"):
        import requests
        url = "http://{}/upload/image".format(self.server_address)
        files = {'image': (name, image_data)}
        response = requests.post(url, files=files)
        return response.json()
    
    def get_images(self, ws, prompt):
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break #Execution is done
                    elif message['type'] == 'execution_error':
                         print(f"ComfyUI Execution Error: {message['data']}")
                         raise Exception(f"ComfyUI Error: {message['data']}")
            except Exception as e:
                print(f"Socket receive error: {e}")
                raise e
            else:
                continue #previews

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
                output_images[node_id] = images_output

        return output_images

    def generate(self, workflow, anchor_bytes, emotion, dramatic_pose=False):
        # 1. Upload Anchor Image
        filename = f"anchor_{self.client_id}.png"
        upload_resp = self.upload_image(anchor_bytes, filename)
        upload_name = upload_resp.get("name")

        # 2. Modify Workflow
        workflow = json.loads(json.dumps(workflow)) # Deep copy
        
        load_image_node = None
        clip_text_node = None
        
        for node_id, node in workflow.items():
            if node["class_type"] == "LoadImage":
                load_image_node = node
            # Try to find a positive prompt Text Encode node
            if (node["class_type"] == "CLIPTextEncode" or node["class_type"] == "BNK_CLIPTextEncode"):
                text = node["inputs"].get("text", "").lower()
                if "negative" not in text and "bad" not in text:
                     clip_text_node = node

        if load_image_node:
             load_image_node["inputs"]["image"] = upload_name
        
        if clip_text_node:
             pose_instruction = ""
             if dramatic_pose:
                 pose_instruction = "dramatic pose, expressive, "
             
             prompt_addition = f"expression: {emotion}, {pose_instruction}"
             
             # Support placeholder or append
             current_text = clip_text_node["inputs"]["text"]
             if "{emotion}" in current_text:
                 clip_text_node["inputs"]["text"] = current_text.replace("{emotion}", emotion)
             else:
                 clip_text_node["inputs"]["text"] = f"{current_text}, {prompt_addition}"

        # 3. Generate
        try:
             images = self.get_images(self.ws, workflow)
             for node_id, image_list in images.items():
                 if image_list:
                     return image_list[0]
        except Exception as e:
             print(f"ComfyUI Error: {e}")
             return None
        return None

def remove_background(image_bytes: bytes) -> bytes:
    """Removes the background from image bytes using rembg."""
    return remove(image_bytes)

async def generate_expression(
    client: genai.Client | ComfyUIBackend,
    anchor_image_bytes: bytes,
    emotion: str,
    output_path: Path,
    model_name: str = "gemini-2.5-flash-image",
    dramatic_pose: bool = False,
    backend_type: str = "gemini",
    workflow: dict = None,
):
    """Sends the anchor image to Gemini or ComfyUI to generate a specific expression."""
    
    if backend_type == "comfyui":
        if not workflow:
            print("Error: ComfyUI backend selected but no workflow provided.")
            return None
        
        # Run in executor to avoid blocking asyncio loop since ComfyUIBackend is synchronous (urllib/websocket)
        loop = asyncio.get_running_loop()
        generated_bytes = await loop.run_in_executor(
            None, 
            client.generate, 
            workflow, 
            anchor_image_bytes, 
            emotion, 
            dramatic_pose
        )
        
        if not generated_bytes:
            print(f"ComfyUI returned no image for {emotion}")
            return None
            
        final_bytes = remove_background(generated_bytes)
        img = Image.open(io.BytesIO(final_bytes))
        if output_path:
            img.save(output_path, format="WEBP")
        return img


    # GEMINI IMPLEMENTATION
    if dramatic_pose:
        pose_instruction = f"Change the pose to be dramatic and expressive to match the {emotion}. Maintain character costume and lighting details."
    else:
        pose_instruction = "Maintain exact character details, pose, and lighting."

    prompt = (
        f"Modify this image to show a {emotion} expression. "
        f"{pose_instruction} "
        "Keep the background a flat, solid color (e.g., plain white or green) so it's easy to remove."
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, types.Part.from_bytes(data=anchor_image_bytes, mime_type="image/png")],
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                        threshold="BLOCK_NONE",
                    ),
                ]
            )
        )

        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            print(f"No content parts returned for {emotion}.")
            return None

        image_part = next((part for part in response.candidates[0].content.parts if part.inline_data), None)
        
        if not image_part:
             print(f"No image part found in response for {emotion}.")
             return None

        generated_bytes = image_part.inline_data.data
        final_bytes = remove_background(generated_bytes)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(final_bytes))
        
        if output_path:
            # Save as WebP
            img.save(output_path, format="WEBP")
        
        return img
            
    except Exception as e:
        print(f"Failed to generate {emotion}: {e}")
        return None

async def run_batch_generation(api_key, anchor_image, name, model_choice, dramatic_pose, backend_choice, comfy_url, workflow_file):
    if backend_choice == "Gemini":
        if not api_key:
            yield [], "Error: API Key is required for Gemini."
            return
        client = genai.Client(api_key=api_key)
        workflow = None
        backend_type = "gemini"
    else:
        # ComfyUI
        if not comfy_url:
            yield [], "Error: ComfyUI URL is required."
            return
        if not workflow_file:
            yield [], "Error: Workflow file (JSON API format) is required for ComfyUI."
            return
        
        try:
             # Load workflow JSON
             # workflow_file is a NamedString or temp file path from Gradio
             with open(workflow_file.name, 'r') as f:
                 workflow = json.load(f)
        except Exception as e:
            yield [], f"Error loading workflow: {e}"
            return

        client = ComfyUIBackend(comfy_url.replace("http://", "").replace("https://", "").replace("/", ""))
        backend_type = "comfyui"

    if anchor_image is None:
        yield [], "Error: Anchor image is required."
        return
    
    output_dir = Path("./output") / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_byte_arr = io.BytesIO()
    anchor_image.save(img_byte_arr, format='PNG')
    anchor_bytes = img_byte_arr.getvalue()

    # Persist API Key if it's new
    env_path = Path(".env")
    current_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != current_key:
        # Create file if it doesn't exist
        if not env_path.exists():
            env_path.touch()
        set_key(env_path, "GOOGLE_API_KEY", api_key)
        os.environ["GOOGLE_API_KEY"] = api_key # Update current session
        print("API Key saved to .env")

    processed = []

    processed = []
    consecutive_failures = 0
    max_failures = 3

    import traceback
    try:
        for emotion in EMOTIONS:
            if consecutive_failures >= max_failures:
                print(f"Stopping generation: {max_failures} consecutive failures.")
                yield processed, f"Stopped: {max_failures} consecutive failures (check logs for errors)."
                break

            print(f"Processing {emotion}...")
            output_file = output_dir / f"{emotion.lower()}.webp"
            
            if output_file.exists():
                print(f"Skipping {emotion} (already exists)")
                yield processed, f"Skipping {emotion} (already exists)... ({len(processed)}/{len(EMOTIONS)})"
                # Load existing image to show in gallery
                try:
                    existing_img = Image.open(output_file)
                    processed.append(existing_img)
                except:
                    pass
                continue

            img = await generate_expression(
                client, 
                anchor_bytes, 
                emotion, 
                output_file, 
                model_choice, 
                dramatic_pose, 
                backend_type=backend_type,
                workflow=workflow
            )
            
            if img:
                processed.append(img)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            
            # Yield current list of images and status
            yield processed, f"Generating... ({len(processed)}/{len(EMOTIONS)})"
            
            # Rate limiting
            time.sleep(10) 
            
        yield processed, "Done!"

    except Exception as e:
        print(f"CRITICAL ERROR in run_batch_generation: {e}")
        traceback.print_exc()
        yield processed, f"Error: {e}"

@app.command()
def generate(
    anchor: Path = typer.Argument(..., help="Path to the anchor image"),
    name: str = typer.Option(..., "--name", "-n", help="Character name (folder name)"),
    api_key: Optional[str] = typer.Option(None, envvar="GOOGLE_API_KEY", help="Google GenAI API Key"),
    model: str = typer.Option("gemini-2.5-flash-image", "--model", help="Gemini model to use"),
    dramatic: bool = typer.Option(False, "--dramatic", "-d", help="Enable dramatic poses matching the emotion"),
    backend: str = typer.Option("gemini", "--backend", "-b", help="Backend to use: gemini or comfyui"),
    comfy_url: str = typer.Option("127.0.0.1:8188", "--comfy-url", help="ComfyUI URL"),
    workflow: Optional[Path] = typer.Option(None, "--workflow", "-w", help="Path to ComfyUI Workflow API JSON"),
):
    """
    Generate SillyTavern expression sprites from a single anchor image.
    """
    if not anchor.exists():
        typer.echo(f"Error: Anchor image not found at {anchor}")
        raise typer.Exit(1)

    backend_choice = "Gemini" if backend.lower() == "gemini" else "ComfyUI"
    
    # Mock file object for run_batch_generation if workflow path is provided
    workflow_file = None
    if workflow:
        if not workflow.exists():
             typer.echo(f"Error: Workflow file not found at {workflow}")
             raise typer.Exit(1)
        # Create a simple object with a .name attribute to mimic Gradio file object
        class FileObj:
            def __init__(self, path): self.name = str(path)
        workflow_file = FileObj(workflow)

    if backend_choice == "Gemini" and not api_key:
        typer.echo("Error: Google API Key is required. Set GOOGLE_API_KEY env var or use --api-key.")
        raise typer.Exit(1)

    # Re-using the logic from run_batch_generation is tricky because it yields.
    # We should just copy the setup logic or refactor. 
    # For CLI, we'll just instantiate and call generate_expression directly loop.
    
    if backend_choice == "ComfyUI":
         if not workflow:
              typer.echo("Error: --workflow JSON file is required for ComfyUI.")
              raise typer.Exit(1)
         client = ComfyUIBackend(comfy_url.replace("http://", "").replace("https://", "").replace("/", ""))
         with open(workflow, 'r') as f:
             workflow_data = json.load(f)
    else:
         client = genai.Client(api_key=api_key)
         workflow_data = None

    output_dir = Path("./output") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(anchor, "rb") as f:
        anchor_bytes = f.read()

    typer.echo(f"Generating sprites for {name} using {backend_choice}...")

    async def process_all():
        for emotion in tqdm(EMOTIONS, desc="Generating Emotions"):
            output_file = output_dir / f"{emotion.lower()}.webp"
            if output_file.exists():
                tqdm.write(f"Skipping {emotion} - already exists.")
                continue
            
            await generate_expression(
                client, 
                anchor_bytes, 
                emotion, 
                output_file, 
                model, 
                dramatic, 
                backend_type=backend.lower(),
                workflow=workflow_data
            )
            time.sleep(5) # Rate limiting

    asyncio.run(process_all())
    typer.echo("Finished! All sprites generated.")

@app.command()
def gui():
    """Launch the Gradio Web UI."""
    with gr.Blocks(title="SillyTavern Sprite Generator") as demo:
        gr.Markdown("# 🎭 SillyTavern Sprite Generator")
        gr.Markdown("Generate all character expressions from a single image using Gemini or Local ComfyUI.")
        
        with gr.Row():
            with gr.Column():
                backend_choice = gr.Radio(
                    choices=["Gemini", "ComfyUI"], 
                    value="Gemini", 
                    label="Backend"
                )
                
                # Gemini Controls
                with gr.Group(visible=True) as gemini_group:
                    api_key = gr.Textbox(
                        label="Google API Key", 
                        placeholder="Paste your GOOGLE_API_KEY here", 
                        type="password", 
                        value=os.getenv("GOOGLE_API_KEY", "")
                    )
                    model_name = gr.Dropdown(
                        choices=["gemini-2.5-flash-image", "gemini-2.0-flash-exp", "gemini-3.0-pro-image-exp"], 
                        value="gemini-2.5-flash-image", 
                        label="Model"
                    )

                # ComfyUI Controls
                with gr.Group(visible=False) as comfy_group:
                    comfy_url = gr.Textbox(
                        label="ComfyUI URL", 
                        value="127.0.0.1:8188", 
                        placeholder="e.g. 127.0.0.1:8188"
                    )
                    workflow_file = gr.File(
                        label="Workflow API JSON", 
                        file_types=[".json"],
                        file_count="single"
                    )
                    gr.Markdown("ℹ️ **Note:** Workflow must be in **API Format** (Enable Dev Mode in ComfyUI -> Save (API Format)). It must have a 'Load Image' node.")

                def toggle_backend(choice):
                    return {
                        gemini_group: gr.update(visible=(choice == "Gemini")),
                        comfy_group: gr.update(visible=(choice == "ComfyUI"))
                    }

                backend_choice.change(
                    fn=toggle_backend,
                    inputs=[backend_choice],
                    outputs=[gemini_group, comfy_group]
                )

                char_name = gr.Textbox(label="Character Name", placeholder="e.g. Vesper")
                dramatic_pose = gr.Checkbox(label="Enable Dramatic Poses", value=False)
                anchor_img = gr.Image(label="Anchor Image", type="pil")
                
                btn = gr.Button("🚀 Generate Sprites", variant="primary")
                stop_btn = gr.Button("🛑 Stop", variant="stop")
                status_msg = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                gallery = gr.Gallery(label="Generated Sprites", columns=4, height="auto")
        
        generate_event = btn.click(
            fn=run_batch_generation, 
            inputs=[
                api_key, 
                anchor_img, 
                char_name, 
                model_name, 
                dramatic_pose,
                backend_choice,
                comfy_url,
                workflow_file
            ], 
            outputs=[gallery, status_msg]
        )
        
        stop_btn.click(fn=None, cancels=[generate_event])
        
    demo.queue().launch()

if __name__ == "__main__":
    app()
