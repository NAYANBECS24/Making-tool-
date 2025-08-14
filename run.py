# run.py
import threading
from interface import FileMaskingApp
import tkinter as tk

if __name__ == '__main__':
    root = tk.Tk()
    app = FileMaskingApp(root)
    root.mainloop()