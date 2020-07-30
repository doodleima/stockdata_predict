import tkinter

from main import search

window = tkinter.Tk()

window.title("Stock Investment")
window.geometry("320x250+100+100")
window.resizable(False, False)

def jongmok(event):
    text = entry.get()
    title.config(text=str(text))
    text1, text2, text3 = search(text) # text3
    switch(text1, text2, text3) # text3

def switch(text1, text2, text3) : # text3
    mess = tkinter.Message(window, text=text1, width=500)
    mess.place(x=60, y=100)

    mess2 = tkinter.Message(window, text=text2, width=1000)
    mess2.place(x=55, y=140)

    mess3 = tkinter.Message(window, text=text3, width=300)
    mess3.place(x=110, y=170)

title = tkinter.Label(window, text = "종목 이름 입력 후 Enter")
title.pack()

entry = tkinter.Entry(window)
entry.bind("<Return>", jongmok)
entry.pack(side = 'top')

# x, y 위치 조정용
#mess3 = tkinter.Message(window, text="정확도 : 0.57", width=300)
#mess3.place(x=110, y=170)

window.mainloop()
