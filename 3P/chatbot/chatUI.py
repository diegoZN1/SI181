from tkinter import (
    DISABLED,
    END,
    FALSE,
    NORMAL,
    Button,
    Scrollbar,
    Tk,
    Text,
)

from chat import get_chatbot_response


def _send():
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg != "":
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + "\n\n")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = get_chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + "\n\n")

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


if __name__ == "__main__":
    base = Tk()
    base.title("Chatbot")
    base.geometry("400x500")
    base.resizable(width=FALSE, height=FALSE)

    ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
    ChatLog.config(state=DISABLED)

    scroll = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog["yscrollcommand"] = scroll.set

    sendbutton = Button(
        base,
        font=("Verdana", 12, "bold"),
        text="Send",
        width="12",
        height=5,
        bd=0,
        bg="#32de97",
        activebackground="#3c9d9b",
        fg="#ffffff",
        command=_send,
    )

    EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
    scroll.place(x=376, y=6, height=386)
    ChatLog.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    sendbutton.place(x=6, y=401, height=90)
    base.mainloop()
