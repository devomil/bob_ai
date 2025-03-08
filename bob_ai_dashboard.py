import tkinter as tk
from tkinter import messagebox, ttk
import os
import psutil  # For system health stats


class BOBDashboard:
    """
    GUI Dashboard for BOB AI.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("BOB AI Dashboard")
        self.root.geometry("500x400")

        # Track proactive suggestions state
        self.proactive_suggestions_enabled = tk.BooleanVar(value=True)

        # System Health Section
        self.system_health_frame = ttk.Frame(self.root)
        self.system_health_frame.pack(pady=10)

        self.memory_label = ttk.Label(self.system_health_frame, text="Memory Usage: --")
        self.memory_label.pack()

        self.cpu_label = ttk.Label(self.system_health_frame, text="CPU Usage: --")
        self.cpu_label.pack()

        # Execution Feedback Section
        self.feedback_frame = ttk.Frame(self.root)
        self.feedback_frame.pack(pady=10)

        self.feedback_label = ttk.Label(self.feedback_frame, text="Execution Feedback:")
        self.feedback_label.pack()

        self.feedback_text = tk.Text(self.feedback_frame, height=10, width=60, state=tk.DISABLED)
        self.feedback_text.pack()

        # Interactive Buttons Section
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=10)

        # "View Created Modules" Button
        self.view_modules_button = ttk.Button(self.button_frame, text="View Created Modules", command=self.view_created_modules)
        self.view_modules_button.pack(side=tk.LEFT, padx=5)

        # "Restart BOB" Button
        self.restart_button = ttk.Button(self.button_frame, text="Restart BOB", command=self.restart_bob)
        self.restart_button.pack(side=tk.LEFT, padx=5)

        # "Enable Proactive Suggestions" Checkbox
        self.suggestions_checkbox = ttk.Checkbutton(
            self.button_frame,
            text="Enable Proactive Suggestions",
            variable=self.proactive_suggestions_enabled
        )
        self.suggestions_checkbox.pack(side=tk.LEFT, padx=5)

        # Start real-time system health updates
        self.update_system_health()

    def display_feedback(self, feedback):
        """
        Display feedback in the feedback text box.
        """
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.insert(tk.END, feedback + "\n")
        self.feedback_text.config(state=tk.DISABLED)
        self.feedback_text.see(tk.END)

    def update_system_health(self):
        """
        Update the system health display with current memory and CPU usage.
        """
        memory = psutil.virtual_memory()
        memory_usage = (memory.used / memory.total) * 100
        cpu_usage = psutil.cpu_percent(interval=0.5)

        self.memory_label.config(text=f"Memory Usage: {memory_usage:.1f}% ({memory.used // (1024 ** 2)}MB / {memory.total // (1024 ** 2)}MB)")
        self.cpu_label.config(text=f"CPU Usage: {cpu_usage:.1f}%")

        # Schedule the next update
        self.root.after(1000, self.update_system_health)

    def view_created_modules(self):
        """
        Display a list of dynamically created modules.
        """
        modules_dir = "core"
        module_files = [f for f in os.listdir(modules_dir) if f.endswith(".py")]

        if module_files:
            messagebox.showinfo("Created Modules", "\n".join(module_files))
        else:
            messagebox.showinfo("Created Modules", "No modules have been created yet.")

    def restart_bob(self):
        """
        Restart BOB by restarting the script.
        """
        self.display_feedback("Restarting BOB...")
        python = os.sys.executable
        os.execl(python, python, "-m", "core.bob_ai")

    def is_proactive_suggestions_enabled(self):
        """
        Check if proactive suggestions are enabled.
        """
        return self.proactive_suggestions_enabled.get()
