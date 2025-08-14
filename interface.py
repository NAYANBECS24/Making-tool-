import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import requests
import os
import time
import threading
from waitress import serve
from app import app

class FileMaskingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PII Masking Tool")
        self.root.geometry("1000x700")
        
        self.api_url = "http://127.0.0.1:5000"
        self.max_retries = 5
        self.retry_delay = 1
        self.download_link = None
        
        self.create_widgets()
        self.start_flask_backend()
        self.root.after(100, self.check_backend_health)

    def start_flask_backend(self):
        """Start Flask server in a separate thread"""
        self.flask_thread = threading.Thread(
            target=lambda: serve(app, host="127.0.0.1", port=5000),
            daemon=True
        )
        self.flask_thread.start()

    def check_backend_health(self):
        """Check if backend is ready"""
        for i in range(self.max_retries):
            try:
                response = requests.get(f"{self.api_url}/health", timeout=2)
                if response.status_code == 200:
                    self.status_var.set("Ready to process files")
                    self.toggle_ui_state(True)
                    return True
            except requests.exceptions.RequestException:
                self.status_var.set(f"Connecting to backend... (Attempt {i+1}/{self.max_retries})")
                self.root.update()
                time.sleep(self.retry_delay)
        
        self.status_var.set("Backend connection failed")
        self.toggle_ui_state(False)
        messagebox.showerror(
            "Connection Error",
            "Could not connect to processing server.\nPlease check if the backend is running."
        )
        return False

    def toggle_ui_state(self, enabled: bool):
        """Enable/disable UI controls based on connection state"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.process_btn.config(state=state)
        self.download_btn.config(state=tk.DISABLED)  # Initially disabled
        self.browse_btn.config(state=state)

    def create_widgets(self):
        """Create all UI components"""
        # Upload Section
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
        
        # Results Display
        results_frame = ttk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Original content
        orig_frame = ttk.LabelFrame(results_frame, text="Original Content")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_text = scrolledtext.ScrolledText(orig_frame, wrap=tk.WORD, height=20)
        self.original_text.pack(fill=tk.BOTH, expand=True)
        
        # Masked content / Detection results
        masked_frame = ttk.LabelFrame(results_frame, text="Masked Content / Detections")
        masked_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.masked_text = scrolledtext.ScrolledText(masked_frame, wrap=tk.WORD, height=20)
        self.masked_text.pack(fill=tk.BOTH, expand=True)
        
        # Download Button
        self.download_btn = ttk.Button(
            self.root,
            text="Download Masked File",
            command=self.download_result,
            state=tk.DISABLED
        )
        self.download_btn.pack(side=tk.BOTTOM, pady=5)
        
        # Retry Button
        self.retry_btn = ttk.Button(
            self.root,
            text="Retry Connection",
            command=self.retry_connection
        )
        self.retry_btn.pack(side=tk.BOTTOM, pady=5)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Starting application...")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, padx=10, pady=5)

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

    def process_file(self):
        filepath = self.file_path.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid file")
            return
        
        try:
            self.status_var.set("Processing file...")
            self.download_btn.config(state=tk.DISABLED)  # Disable until processing completes
            self.root.update()
            
            with open(filepath, 'rb') as f:
                files = {'file': (os.path.basename(filepath), f)}
                response = requests.post(f"{self.api_url}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"API response: {result}")  # Debug logging
                self.display_results(result)
                self.download_link = result.get('download_link')
                self.status_var.set("File processed successfully")
            else:
                messagebox.showerror("Error", f"API Error: {response.json().get('error', 'Unknown error')}")
                self.status_var.set("Error processing file")
                self.download_link = None
                self.download_btn.config(state=tk.DISABLED)
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.download_link = None
            self.download_btn.config(state=tk.DISABLED)

    def display_results(self, result):
        self.original_text.delete(1.0, tk.END)
        self.original_text.insert(tk.END, result.get('original_text', ''))
        
        self.masked_text.delete(1.0, tk.END)
        if result.get('entities') is not None:  # Image-based detection
            entities = result.get('entities', [])
            output = "Detected PII (Bounding Boxes):\n"
            if not entities:
                output += "No PII detected."
            for entity in entities:
                output += f"Type: {entity['type']}, Score: {entity['score']:.2f}, BBox: {entity['bbox']}\n"
            self.masked_text.insert(tk.END, output)
            self.download_btn.config(state=tk.NORMAL if result.get('download_link') else tk.DISABLED)
        else:  # Text-based masking
            self.masked_text.insert(tk.END, result.get('masked_text', ''))
            self.download_btn.config(state=tk.NORMAL if result.get('download_link') else tk.DISABLED)

    def download_result(self):
        if not self.download_link:
            messagebox.showwarning("Warning", "No masked file to download")
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
                response = requests.get(download_url)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                messagebox.showinfo("Success", f"File saved to {save_path}")
                self.status_var.set(f"File saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Download failed: {str(e)}")
                self.status_var.set("Download failed")

    def retry_connection(self):
        """Handle retry connection button"""
        self.retry_btn.config(state=tk.DISABLED)
        self.status_var.set("Attempting to reconnect...")
        self.root.update()
        
        if self.check_backend_health():
            messagebox.showinfo("Success", "Successfully connected to backend")
            self.retry_btn.config(state=tk.NORMAL)
        else:
            self.retry_btn.config(state=tk.NORMAL)

if __name__ == '__main__':
    root = tk.Tk()
    app = FileMaskingApp(root)
    root.mainloop()