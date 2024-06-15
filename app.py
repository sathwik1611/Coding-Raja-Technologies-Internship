# Importing required libraries
from tkinter import *
from chat import get_response, bot_name

# GUI Color
BG_BLUE = '#4682B4'
BG_COLOR = '#17202A'
TEXT_COLOR = '#00FFFF'

# Font
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


# Application class
class ChatApplication:
    # Creating GUI Window
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()  # helper function

    # Run our application
    def run(self):
        self.window.mainloop()

    # setup main window
    def _setup_main_window(self):
        self.window.title("ChatBot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # Layout
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # Tiny divider
        line = Label(self.window, width=450, bg=BG_BLUE)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=12, bg=BG_COLOR,
                                fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # Scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # Bottom label
        bottom_label = Label(self.window, bg=BG_COLOR, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.001)
        self.msg_entry.focus()  # when the ap gets started this gets the focus and we can type the input
        self.msg_entry.bind("<Return>", self._on_enter_pressed)  # To send the message to the function

        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_BLUE,
                             command=lambda : self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    # Message received and applying function
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()  # get the input text
        self._insert_message(msg, "You")

    # to insert in text area
    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)  # to delete the entry message
        msg1 = f'{sender}: {msg}\n\n'
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=NORMAL)

        # Response from the bot
        msg2 = f'{bot_name}: {get_response(msg)}\n\n'
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=NORMAL)

        self.text_widget.see(END)


if __name__ == "__main__":
    app = ChatApplication()
    app.run()
