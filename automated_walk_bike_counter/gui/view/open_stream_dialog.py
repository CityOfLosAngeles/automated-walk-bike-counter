from tkinter import (
    Button,
    E,
    Entry,
    Frame,
    Label,
    N,
    S,
    StringVar,
    Toplevel,
    W,
    messagebox,
)

from ...model.video.video import VideoStream


class OpenStreamDialog:
    def __init__(self, parent, controller):
        top = self.top = Toplevel(parent)
        self.controller = controller
        self.stream_url = StringVar()
        base_frame = Frame(top)
        base_frame.grid(
            row=0, column=0, padx=(10, 10), pady=(15, 15), sticky=(W, E, S, N)
        )
        self.label1 = Label(master=base_frame, text="Please enter your stream URL:")
        self.label1.grid(row=0, column=0, pady=(0, 5), sticky=W)
        self.entry1 = Entry(master=base_frame, textvariable=self.stream_url, width=70)
        self.entry1.grid(row=1, column=0, pady=(0, 5), sticky=(W, E))
        self.open_btn = Button(
            base_frame, text="Open", width=20, command=self.open_stream
        )
        self.open_btn.grid(row=2, column=0, pady=(15, 5))

        parent_left = parent.winfo_rootx()
        parent_top = parent.winfo_rooty()

        top.protocol("WM_DELETE_WINDOW")
        top.geometry("450x150+%d+%d" % (parent_left + 100, parent_top + 100))

        top.focus_set()
        top.grab_set()
        self.entry1.focus_set()
        top.wait_window()

    def open_stream(self):
        if self.stream_url.get() == "":
            messagebox.showwarning("Warning", "Enter a valid address for stream video!")
            return
        self.controller.stream = VideoStream(self.stream_url.get())
        self.controller.input_camera_type = "webcam"
        self.top.destroy()
