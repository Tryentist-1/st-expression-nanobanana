# How to Create a Mac App for Silly Sprites

You can create a standalone "Silly Sprites" app that sits in your Dock and launches with one click.

## Step 1: Create the App with Automator

1.  Open **Automator** (cmd+space, type "Automator").
2.  Choose **Application** as the document type.
3.  In the search bar, type `Run Shell Script` and drag it to the right workflow area.
4.  **Important**: Change "Pass input" to **as arguments**.
5.  Paste the following code into the box (replace the text that's there):

    ```bash
    # Fix path for typical Mac setups
    export PATH="/Users/terry/.local/bin:$PATH"
    
    # Go to project directory
    cd "/Users/terry/makeitso/silly_sprites"
    
    # Run the app in the background
    # We use nohup to keep it running if the script closes, 
    # but for an Automator app, we want it to stay open or manage the window.
    # The simplest way is:
    
    /Users/terry/.local/bin/uv run sprites.py gui &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 4
    
    # Open Browser
    open http://127.0.0.1:7860
    
    # Keep the script running to maintain the server
    wait $SERVER_PID
    ```

6.  Press `Cmd+S` to save.
7.  Name it **Silly Sprites** and save it to your **Applications** folder (or Desktop).

## Step 2: Add a Custom Icon

1.  Find an image you want to use as an icon (e.g., a `.png` file).
2.  Open the image in **Preview**.
3.  Press `Cmd+A` (Select All) and then `Cmd+C` (Copy).
4.  Find your new **Silly Sprites.app** in Finder.
5.  Right-click it and choose **Get Info** (or `Cmd+I`).
6.  Click the small icon in the **top-left corner** of the Info window (it will highlight).
7.  Press `Cmd+V` (Paste).

Your app now has a custom icon! Drag it to your Dock to pin it.
