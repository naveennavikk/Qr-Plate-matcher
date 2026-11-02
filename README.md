# Reference Plate Matcher

This VS Code project checks whether a reference plate appears in a property image and returns **ACCEPT** or **REJECT**.

## Run in VS Code

1. Extract the ZIP and open the extracted `reference_plate_matcher` folder in VS Code.
2. Open the integrated terminal and create/activate a virtual environment.

   **PowerShell:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Command Prompt:**
   ```bat
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:
   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed in the terminal. Upload the reference plate and property image, then click **Check plate**.

The fast matcher downsizes property photos above 1400 pixels on the long edge and uses a reduced set of scales. It keeps the original image for display and maps match coordinates back to the original resolution. Grounding DINO is only loaded and run as a fallback if the fast search misses; its first fallback use downloads model files from Hugging Face. The `torch` and `torchvision` packages are installed as CPU builds by default; for GPU acceleration, install the appropriate PyTorch build for your system first, then install the remaining packages.
