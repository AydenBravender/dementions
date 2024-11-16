import tkinter as tk

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Touchscreen Drawing App")
        
        # Create a canvas to draw on
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        # Variables for storing previous touch position
        self.last_x = None
        self.last_y = None

        # Bind touch events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.canvas.bind("<ButtonPress-1>", self.store_coords)

    def paint(self, event):
        """Draw on the canvas when the mouse is moved (or finger dragged on touch)."""
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=2, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x = event.x
        self.last_y = event.y

    def store_coords(self, event):
        """Store the initial coordinates when the user presses the mouse (or starts touch)."""
        self.last_x = event.x
        self.last_y = event.y

    def reset(self, event):
        """Reset the coordinates when the mouse button is released (or touch ends)."""
        self.last_x = None
        self.last_y = None

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
