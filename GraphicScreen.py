from tkinter import *

root = Tk()

root.title("Welcome to our fracture detection site.")
lbl = Label(root, text = "What type of fracture do you want to detect")
lbl.grid()

root.geometry('350x200')

def clicked():
    lbl.configure()

btn = Button(root, text = "Click me" ,
             fg = "red", command = clicked)
btn.grid(column=2, row=0)
root.mainloop()

