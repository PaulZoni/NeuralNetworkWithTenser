from tkinter import *


class CanvasDemo:
    def __init__(self):
        window = Tk()
        window.title("Canvas Demo")

        self.canvas = Canvas(window, width=340, height=340,
                             bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.processMouseEvent)
        self.board = self.makeVirtualBoard()

        frame = Frame(window)
        frame.pack()
        btMakeBoard = Button(frame, text="Reset Board", command=self.makeBoardPart1)
        btMakeBoard.grid(row=1, column=1)

        window.mainloop()

    def makeBoardPart1(self):
        self.canvas.create_rectangle(10, 10, 330, 330, tags="outline")
        y = 10
        x = 10
        while x < 320:
            while y < 320:
                self.canvas.create_rectangle(x, y, x + 40, y + 40, fill="black", tags="squares")
                y += 80
            y = 10
            x += 80
        self.makeBoardPart2()
        self.setPieces()

    def makeBoardPart2(self):
        y = 50
        x = 50
        while x < 320:
            while y < 320:
                self.canvas.create_rectangle(x, y, x + 40, y + 40, fill="black", tags="squares")
                y += 80
            y = 50
            x += 80

    def setPieces(self):
        x1 = 10
        while x1 < 320:
            self.canvas.create_oval(x1, 10, x1 + 40, 50, fill="red", tags="red")
            x1 += 80

        x1 = 10
        while x1 < 320:
            self.canvas.create_oval(x1, 90, x1 + 40, 130, fill="red", tags="red")
            x1 += 80

        x1 = 50
        while x1 < 320:
            self.canvas.create_oval(x1, 50, x1 + 40, 90, fill="red", tags="red")
            x1 += 80

        x1 = 50
        while x1 < 320:
            self.canvas.create_oval(x1, 210, x1 + 40, 250, fill="white", tags="white")
            x1 += 80

        x1 = 50
        while x1 < 320:
            print(self.canvas.create_oval(x1, 290, x1 + 40, 330, fill="white", tags="white"))
            x1 += 80

        x1 = 10
        while x1 < 320:
            self.canvas.create_oval(x1, 250, x1 + 40, 290, fill="white", tags="white")
            x1 += 80

    def processMouseEvent(self, event):
        print(event.x, event.y)
        print(event.x_root, event.y_root)
        print(event.num)
        self.check_event(event)

    def check_event(self, event):
        self.canvas.delete(53)
        if event.x < 40 and event.y > 300:
            print('A:8')
            self.canvas.create_oval(event.x, event.y, event.x + 10, event.y + 10, fill="white", tags="white")
        if 80 > event.x > 40 and 350 > event.y > 300:
            print('B:8')
            self.canvas.delete(int(self.board[7][1]['id']))

    def makeVirtualBoard(self):
        virtualBoard = [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [{'A:8': 0, 'id': None}, {'B:8': 1, 'id': 50}, {'C:8': 0, 'id': None}, {'D:8': 1, 'id': 51}, {'E:8': 0, 'id': None}, {'F:8': 1, 'id': 52}, {'G:8': 0, 'id': None}, {'H:8': 1, 'id': 53}],
        ]
        return virtualBoard

CanvasDemo()  # Create GUI