import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Toplevel
import requests
import os
import time
import threading
import logging
from waitress import serve
from app import app
from PIL import Image, ImageTk
import cv2
import io
import numpy as np

try:
    import ttkbootstrap as ttkb
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False
    print("ttkbootstrap not found; falling back to standard ttk theme")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileMaskingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ID Masker Tool")
        self.root.geometry("1200x800")
        
        if BOOTSTRAP_AVAILABLE:
            self.root.style = ttkb.Style(theme="darkly")
        else:
            self.root.style = ttk.Style()
            self.root.style.theme_use('clam')

        self.api_url = "http://127.0.0.1:5000"
        self.max_retries = 5
        self.retry_delay = 1
        self.download_link = None
        self.camera_running = False
        self.camera_label = None
        self.captured_image_path = None
        self.feed_thread = None
        self.progress = None
        self.masked_image_label = None
        
        self.create_widgets()
        self.start_flask_backend()
        self.root.after(100, self.check_backend_health)

    def start_flask_backend(self):
        try:
            self.flask_thread = threading.Thread(
                target=lambda: serve(app, host="127.0.0.1", port=5000),
                daemon=True
            )
            self.flask_thread.start()
            logger.info("Flask backend thread started")
        except Exception as e:
            logger.error(f"Failed to start Flask backend: {e}")
            self.status_var.set(f"Failed to start backend: {e}")
            messagebox.showerror("Error", f"Failed to start backend: {e}")

    def check_backend_health(self):
        for i in range(self.max_retries):
            try:
                response = requests.get(f"{self.api_url}/health", timeout=2)
                if response.status_code == 200:
                    self.status_var.set("Ready to process files and camera")
                    self.toggle_ui_state(True)
                    logger.info("Backend health check passed")
                    return True
            except requests.exceptions.RequestException as e:
                self.status_var.set(f"Connecting to backend... (Attempt {i+1}/{self.max_retries})")
                self.root.update()
                logger.warning(f"Backend health check failed: {e}")
                time.sleep(self.retry_delay)
        
        self.status_var.set("Backend connection failed")
        self.toggle_ui_state(False)
        messagebox.showerror("Connection Error", "Could not connect to processing server.")
        logger.error("Backend connection failed after retries")
        return False

    def toggle_ui_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.process_btn.config(state=state)
        self.download_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=state)
        self.camera_btn.config(state=state)
        self.capture_btn.config(state=tk.DISABLED if not self.camera_running else tk.NORMAL)
        logger.debug(f"UI state toggled: {'enabled' if enabled else 'disabled'}")

    def create_widgets(self):
        upload_frame = ttk.LabelFrame(self.root, text="Upload File", padding=10)
        upload_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.file_path = tk.StringVar()
        ttk.Entry(upload_frame, textvariable=self.file_path, width=70).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )
        self.browse_btn = ttk.Button(upload_frame, text="Browse", command=self.browse_file)
        self.browse_btn.pack(side=tk.LEFT)
        
        self.process_btn = ttk.Button(upload_frame, text="Process File", command=self.process_file)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.camera_btn = ttk.Button(upload_frame, text="Start Live Camera", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(upload_frame, text="Capture Image", command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.mask_style = tk.StringVar(value="black_box")
        ttk.Combobox(upload_frame, textvariable=self.mask_style, values=["black_box", "blur", "pixelation"]).pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(upload_frame, orient="horizontal", length=200, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, padx=5, pady=5)
        self.progress.stop()
        
        results_frame = ttk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        orig_frame = ttk.LabelFrame(results_frame, text="Original Content")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_text = scrolledtext.ScrolledText(orig_frame, wrap=tk.WORD, height=20)
        self.original_text.pack(fill=tk.BOTH, expand=True)
        
        masked_frame = ttk.LabelFrame(results_frame, text="Detected PII")
        masked_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.masked_text = scrolledtext.ScrolledText(masked_frame, wrap=tk.WORD, height=20)
        self.masked_text.pack(fill=tk.BOTH, expand=True)
        
        camera_frame = ttk.LabelFrame(self.root, text="Live Camera Feed")
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack()
        
        masked_image_frame = ttk.LabelFrame(self.root, text="Masked Image")
        masked_image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=5)
        self.masked_image_label = ttk.Label(masked_image_frame)
        self.masked_image_label.pack()
        
        self.download_btn = ttk.Button(
            self.root,
            text="Download Masked File",
            command=self.download_result,
            state=tk.DISABLED
        )
        self.download_btn.pack(side=tk.BOTTOM, pady=5)
        
        self.retry_btn = ttk.Button(
            self.root,
            text="Retry Connection",
            command=self.retry_connection
        )
        self.retry_btn.pack(side=tk.BOTTOM, pady=5)
        
        self.status_var = tk.StringVar(value="Starting application...")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, padx=10, pady=5)
        
        if BOOTSTRAP_AVAILABLE:
            self.add_tooltips()

    def add_tooltips(self):
        self.create_tooltip(self.browse_btn, "Browse for image, PDF, or text file")
        self.create_tooltip(self.process_btn, "Process the selected file for PII masking")
        self.create_tooltip(self.camera_btn, "Start/stop live camera feed")
        self.create_tooltip(self.capture_btn, "Capture and process a frame")
        self.create_tooltip(self.download_btn, "Download the masked file")

    def create_tooltip(self, widget, text):
        tooltip = Toplevel(self.root)
        tooltip.withdraw()
        tooltip.wm_overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="yellow", relief='solid', borderwidth=1, padx=5, pady=3)
        label.pack()
        def enter(event):
            try:
                x, y, _, _ = widget.bbox("insert")
                x += widget.winfo_rootx() + 25
                y += widget.winfo_rooty() + 25
                tooltip.wm_geometry(f"+{x}+{y}")
                tooltip.deiconify()
            except Exception as e:
                logger.error(f"Tooltip error: {e}")
        def leave(event):
            tooltip.withdraw()
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def browse_file(self):
        filetypes = (
            ("All supported files", "*.png *.jpg *.jpeg *.pdf *.txt"),
            ("Image files", "*.png *.jpg *.jpeg"),
            ("PDF files", "*.pdf"),
            ("Text files", "*.txt")
        )
        filename = filedialog.askopenfilename(title="Select file", filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            self.status_var.set(f"Selected: {os.path.basename(filename)}")
            logger.info(f"Selected file: {filename}")

    def process_file(self):
        filepath = self.file_path.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid file")
            logger.error("No valid file selected for processing")
            return
        try:
            self.status_var.set("Processing file...")
            self.download_btn.config(state=tk.DISABLED)
            self.progress.start()
            self.root.update()
            with open(filepath, 'rb') as f:
                files = {'file': (os.path.basename(filepath), f)}
                data = {'mask_style': self.mask_style.get()}
                response = requests.post(f"{self.api_url}/upload", files=files, data=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                logger.info(f"API response: {result}")
                self.display_results(result)
                self.download_link = result.get('download_link')
                self.status_var.set("File processed successfully")
                if result.get('entities') and self.download_link:
                    self.display_masked_image(f"{self.api_url}{self.download_link}")
            else:
                messagebox.showerror("Error", f"API Error: {response.json().get('error', 'Unknown error')}")
                self.status_var.set("Error processing file")
                self.download_link = None
                self.download_btn.config(state=tk.DISABLED)
                logger.error(f"API error during file processing: {response.json()}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.download_link = None
            self.download_btn.config(state=tk.DISABLED)
            logger.error(f"File processing error: {e}")
        finally:
            self.progress.stop()

    def capture_image(self):
        if not self.camera_running:
            messagebox.showerror("Error", "Camera is not running")
            logger.error("Capture attempted while camera is not running")
            return
        try:
            self.status_var.set("Capturing image...")
            self.progress.start()
            self.root.update()
            response = requests.post(f"{self.api_url}/capture_image", data={'mask_style': self.mask_style.get()}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                self.captured_image_path = result.get('filepath')
                self.download_link = result.get('download_link')
                self.status_var.set(f"Image captured: {os.path.basename(self.captured_image_path)}")
                logger.info(f"Image captured: {self.captured_image_path}")
                self.file_path.set(self.captured_image_path)
                self.display_results(result)
                if self.download_link:
                    self.display_masked_image(f"{self.api_url}{self.download_link}")
            else:
                messagebox.showerror("Error", f"Failed to capture image: {response.json().get('error', 'Unknown error')}")
                self.status_var.set("Error capturing image")
                logger.error(f"Capture image API error: {response.json()}")
        except Exception as e:
            messagebox.showerror("Error", f"Capture error: {str(e)}")
            self.status_var.set(f"Capture error: {str(e)}")
            logger.error(f"Capture error: {e}")
        finally:
            self.progress.stop()

    def display_results(self, result):
        self.original_text.delete(1.0, tk.END)
        self.original_text.insert(tk.END, result.get('original_text', ''))
        self.masked_text.delete(1.0, tk.END)
        if result.get('entities') is not None:
            entities = result.get('entities', [])
            output = "Detected PII (Bounding Boxes):\n"
            if not entities:
                output += "No PII detected."
            for entity in entities:
                output += f"Type: {entity['type']}, Score: {entity['score']:.2f}, BBox: {entity['bbox']}\n"
            self.masked_text.insert(tk.END, output)
            self.download_btn.config(state=tk.NORMAL if result.get('download_link') else tk.DISABLED)
            logger.info(f"Displayed image detection results: {len(entities)} entities")
        else:
            self.masked_text.insert(tk.END, result.get('masked_text', ''))
            self.download_btn.config(state=tk.NORMAL if result.get('download_link') else tk.DISABLED)
            logger.info("Displayed text/PDF detection results")

    def display_masked_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((320, 240), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.masked_image_label.config(image=photo)
                self.masked_image_label.image = photo
                logger.info("Displayed masked image")
            else:
                logger.error(f"Failed to fetch masked image: HTTP {response.status_code}")
                self.status_var.set(f"Error displaying masked image: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error displaying masked image: {e}")
            self.status_var.set(f"Error displaying masked image: {e}")

    def toggle_camera(self):
        if not self.camera_running:
            try:
                response = requests.post(f"{self.api_url}/start_camera", timeout=5)
                if response.status_code == 200:
                    self.camera_running = True
                    self.camera_btn.config(text="Stop Live Camera")
                    self.capture_btn.config(state=tk.NORMAL)
                    self.status_var.set("Live camera started")
                    logger.info("Camera started successfully")
                    self.feed_thread = threading.Thread(target=self.update_camera_feed, daemon=True)
                    self.feed_thread.start()
                else:
                    messagebox.showerror("Error", f"Failed to start camera: {response.json().get('error', 'Unknown error')}")
                    self.status_var.set("Error starting camera")
                    logger.error(f"Start camera API error: {response.json()}")
            except Exception as e:
                messagebox.showerror("Error", f"Camera error: {str(e)}")
                self.status_var.set(f"Camera error: {str(e)}")
                logger.error(f"Camera start error: {e}")
        else:
            try:
                response = requests.post(f"{self.api_url}/stop_camera", timeout=5)
                if response.status_code == 200:
                    self.camera_running = False
                    self.camera_btn.config(text="Start Live Camera")
                    self.capture_btn.config(state=tk.DISABLED)
                    self.status_var.set("Live camera stopped")
                    self.camera_label.config(image='')
                    self.masked_image_label.config(image='')
                    self.feed_thread = None
                    logger.info("Camera stopped successfully")
                else:
                    messagebox.showerror("Error", f"Failed to stop camera: {response.json().get('error', 'Unknown error')}")
                    self.status_var.set("Error stopping camera")
                    logger.error(f"Stop camera API error: {response.json()}")
            except Exception as e:
                messagebox.showerror("Error", f"Camera error: {str(e)}")
                self.status_var.set(f"Camera error: {str(e)}")
                logger.error(f"Camera stop error: {e}")

    def update_camera_feed(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            self.status_var.set("Failed to open camera")
            self.root.after(0, self.toggle_camera)
            return
        
        while self.camera_running:
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read camera frame")
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((320, 240), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.camera_label.config(image=photo)
                self.camera_label.image = photo
                self.root.update_idletasks()
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                logger.error(f"Camera feed error: {e}")
                self.status_var.set(f"Camera feed error: {e}")
                break
        
        cap.release()
        if self.camera_running:
            self.root.after(0, self.toggle_camera)
        logger.info("Camera feed stopped")

    def download_result(self):
        if not self.download_link:
            messagebox.showwarning("Warning", "No masked file to download")
            logger.warning("Download attempted with no file")
            return
        download_url = f"{self.api_url}{self.download_link}"
        ext = os.path.splitext(self.download_link)[1]
        defaultextension = ext if ext else ".txt"
        filetypes = [("All files", "*.*")]
        if ext == '.png':
            filetypes = [("PNG files", "*.png"), ("All files", "*.*")]
        elif ext == '.txt':
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        save_path = filedialog.asksaveasfilename(
            title="Save masked file",
            defaultextension=defaultextension,
            filetypes=filetypes
        )
        if save_path:
            try:
                response = requests.get(download_url, timeout=5)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                messagebox.showinfo("Success", f"File saved to {save_path}")
                self.status_var.set(f"File saved to {save_path}")
                logger.info(f"File downloaded: {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Download failed: {str(e)}")
                self.status_var.set("Download failed")
                logger.error(f"Download error: {e}")

    def retry_connection(self):
        self.retry_btn.config(state=tk.DISABLED)
        self.status_var.set("Attempting to reconnect...")
        self.root.update()
        if self.check_backend_health():
            messagebox.showinfo("Success", "Successfully connected to backend")
            self.retry_btn.config(state=tk.NORMAL)
            logger.info("Backend reconnection successful")
        else:
            self.retry_btn.config(state=tk.NORMAL)
            logger.error("Backend reconnection failed")

if __name__ == '__main__':
    root = tk.Tk()
    app = FileMaskingApp(root)
    root.mainloop()
