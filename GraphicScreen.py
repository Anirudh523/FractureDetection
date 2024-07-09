from tkinter import *

root = Tk()
colour_frame = Frame(root)
options_frame = Frame(root)
root.title("Welcome to our fracture detection site.")
lbl = Label(root, text = "What type of fracture do you want to detect")

lbl.grid(column = 500, row = 400)


root.geometry('1000x1000')


def clicked():
    lbl.configure()

btn = Button(root, text = "Click me" ,
             fg = "red", command = clicked)
btn.grid(column=100, row=0)
root.mainloop()

