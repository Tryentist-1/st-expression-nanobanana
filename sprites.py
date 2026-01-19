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
from google import genai
from google.genai import types
from dotenv import load_dotenv, set_key

load_dotenv()

app = typer.Typer(help="SillyTavern Sprite Generator")

EMOTIONS = [
    "neutral", "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", 
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", 
    "nervousness", "optimism", "pride", "realization", "relief", "remorse", 
    "sadness", "surprise", "blush"
]

def remove_background(image_bytes: bytes) -> bytes:
    """Removes the background from image bytes using rembg."""
    return remove(image_bytes)

async def generate_expression(
    client: genai.Client,
    anchor_image_bytes: bytes,
    emotion: str,
    output_path: Path,
    model_name: str = "gemini-2.5-flash-image",
    dramatic_pose: bool = False,
):
    """Sends the anchor image to Gemini to generate a specific expression."""
    
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
        # Safety settings to prevent blocking
        # Note: Adjust fields as per the latest SDK spec if needed, but BLOCK_NONE is standard target
        # For google-genai SDK 0.x/1.x, we assume standard config keys.
        
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
            if response.candidates:
                print(f"Finish Reason: {response.candidates[0].finish_reason}")
                print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
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
        try:
            print(f"Response debug: {response}")
        except:
            print("Response not available")
        # traceback.print_exc() # Reduce noise for now
        return None

async def run_batch_generation(api_key, anchor_image, name, model_choice, dramatic_pose):
    if not api_key:
        yield [], "Error: API Key is required."
        return
    if anchor_image is None:
        yield [], "Error: Anchor image is required."
        return
    
    client = genai.Client(api_key=api_key)
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

    for emotion in EMOTIONS:
        if consecutive_failures >= max_failures:
            print(f"Stopping generation: {max_failures} consecutive failures (likely safety refusals).")
            yield processed, f"Stopped: {max_failures} consecutive failures (likely blocked by safety)."
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

        img = await generate_expression(client, anchor_bytes, emotion, output_file, model_choice, dramatic_pose)
        
        if img:
            processed.append(img)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        
        # Yield current list of images and status
        yield processed, f"Generating... ({len(processed)}/{len(EMOTIONS)})"
        
        # Rate limiting
        time.sleep(5) 
        
    yield processed, "Done!"

@app.command()
def generate(
    anchor: Path = typer.Argument(..., help="Path to the anchor image"),
    name: str = typer.Option(..., "--name", "-n", help="Character name (folder name)"),
    api_key: Optional[str] = typer.Option(None, envvar="GOOGLE_API_KEY", help="Google GenAI API Key"),
    model: str = typer.Option("gemini-2.5-flash-image", "--model", help="Gemini model to use"),
    dramatic: bool = typer.Option(False, "--dramatic", "-d", help="Enable dramatic poses matching the emotion"),
):
    """
    Generate SillyTavern expression sprites from a single anchor image.
    """
    if not anchor.exists():
        typer.echo(f"Error: Anchor image not found at {anchor}")
        raise typer.Exit(1)

    if not api_key:
        typer.echo("Error: Google API Key is required. Set GOOGLE_API_KEY env var or use --api-key.")
        raise typer.Exit(1)

    client = genai.Client(api_key=api_key)
    output_dir = Path("./output") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(anchor, "rb") as f:
        anchor_bytes = f.read()

    typer.echo(f"Generating sprites for {name} using {model}...")

    async def process_all():
        for emotion in tqdm(EMOTIONS, desc="Generating Emotions"):
            output_file = output_dir / f"{emotion.lower()}.webp"
            if output_file.exists():
                tqdm.write(f"Skipping {emotion} - already exists.")
                continue
            
            await generate_expression(client, anchor_bytes, emotion, output_file, model, dramatic)
            time.sleep(5) # Rate limiting

    asyncio.run(process_all())
    typer.echo("Finished! All sprites generated.")

@app.command()
def gui():
    """Launch the Gradio Web UI."""
    with gr.Blocks(title="SillyTavern Sprite Generator") as demo:
        gr.Markdown("# 🎭 SillyTavern Sprite Generator")
        gr.Markdown("Generate all character expressions from a single image using Gemini.")
        
        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label="Google API Key", placeholder="Paste your GOOGLE_API_KEY here", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
                char_name = gr.Textbox(label="Character Name", placeholder="e.g. Vesper")
                model_name = gr.Dropdown(
                    choices=["gemini-2.5-flash-image", "gemini-2.0-flash-exp", "gemini-3.0-pro-image-exp"], 
                    value="gemini-2.5-flash-image", 
                    label="Model"
                )
                dramatic_pose = gr.Checkbox(label="Enable Dramatic Poses", value=False)
                anchor_img = gr.Image(label="Anchor Image", type="pil")
                btn = gr.Button("🚀 Generate Sprites", variant="primary")
                stop_btn = gr.Button("🛑 Stop", variant="stop")
                status_msg = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                gallery = gr.Gallery(label="Generated Sprites", columns=4, height="auto")
        
        generate_event = btn.click(
            fn=run_batch_generation, 
            inputs=[api_key, anchor_img, char_name, model_name, dramatic_pose], 
            outputs=[gallery, status_msg]
        )
        
        stop_btn.click(fn=None, cancels=[generate_event])
        
    demo.queue().launch()

if __name__ == "__main__":
    app()
