from PyQt4 import QtGui, QtCore
import cv2
from numpy import interp
import random
import time
import sys
import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.Qt import QTimer, QPixmap, QColor
import neat
import visualize

class Snake:
    def __init__(self, width, height):

        self.maxWidth = width
        self.maxHeight = height

        self.canHitWalls = False

        self.fruitLocationGenerator = np.random.RandomState(random.randint(0, 10000))
        #self.fruitLocationGenerator = np.random.RandomState(100)

        self.alive = True
        self.headX = int(width/2)
        self.headY = int(height/2)

        self.fruitX = self.fruitLocationGenerator.randint(0, width - 1)
        self.fruitY = self.fruitLocationGenerator.randint(0, height - 1)

        while (self.fruitX == self.headX or self.fruitY == self.headY):
            self.fruitX = self.fruitLocationGenerator.randint(0, width - 1)
            self.fruitY = self.fruitLocationGenerator.randint(0, height - 1)

        self.aliveTime = 100
        self.timeLeft = self.aliveTime
        self.score = 0
        self.closeScore = 0


        self.vel = 3 #0 up, 1 right, 2, down, 3 left
        self.previousVel = 3
        self.previousMoves = [3]

    def setVel(self, vel):
        if vel == 3 and self.vel != 1 :
            self.vel = 3
        elif vel == 1 and self.vel != 3 :
            self.vel = 1
        elif vel == 0 and self.vel != 2 :
            self.vel = 0
        elif vel == 2 and self.vel != 0 :
            self.vel = 2


    def update(self):
        if self.alive == False:
            return 
        self.timeLeft -= 1
        self.score -= 1
        if self.timeLeft < 0:
            self.alive = False
            self.score += self.closeScore
            return

        xDirectionVal = .5 / float(self.maxWidth)
        yDirectionVal = .5 / float(self.maxHeight)

        xScore = xDirectionVal * float(abs(self.fruitX - self.headX))
        yScore = yDirectionVal * float(abs(self.fruitY - self.headY))
        self.closeScore = 1 - xScore - yScore

        if self.headX == self.fruitX and self.headY == self.fruitY:
            self.previousMoves.append(self.previousMoves[len(self.previousMoves)-1])
            print "Got a fruit"
            self.fruitX = self.fruitLocationGenerator.randint(0, self.maxWidth- 1)
            self.fruitY = self.fruitLocationGenerator.randint(0, self.maxHeight- 1)
            while (self.fruitX == self.headX or self.fruitY == self.headY):
                self.fruitX = self.fruitLocationGenerator.randint(0, self.maxWidth- 1)
                self.fruitY = self.fruitLocationGenerator.randint(0, self.maxHeight- 1)
            self.score += 100
            self.timeLeft = self.aliveTime 


        if self.vel == 0:
            self.headY -= 1
        elif self.vel == 1:
            self.headX += 1
        elif self.vel == 2:
            self.headY += 1
        elif self.vel == 3:
            self.headX -= 1

        if (self.headX, self.headY) in self.getTailPoints():
            print "Hit itself"
            pass
            #self.alive = False

        self.previousMoves.pop()
        self.previousMoves.insert(0, self.vel)

        if self.headX % self.maxWidth == 0:
            if self.canHitWalls == False:
                self.alive = False
            self.headX = 0
        elif self.headX < 0:
            if self.canHitWalls == False:
                self.alive = False
            self.headX = self.maxWidth - 1

        if self.headY % self.maxHeight == 0:
            if self.canHitWalls == False:
                self.alive = False
            self.headY = 0
        elif self.headY < 0:
            if self.canHitWalls == False:
                self.alive = False
            self.headY = self.maxHeight - 1

        if self.alive == False:
        #    self.score = 0
            pass
            self.score += self.closeScore


    def getTailPoints(self):
        tailPoints = []
        pX = self.headX
        pY = self.headY
        for move in self.previousMoves:
            if move == 0:
                pY += 1
            if move == 1:
                pX -= 1
            if move == 2:
                pY -= 1
            if move == 3:
                pX += 1
            if pX >= self.maxWidth:
                pX = 0
            if pX < 0:
                pX = self.maxWidth - 1
            if pY >= self.maxHeight:
                pY = 0
            if pY < 0:
                pY = self.maxHeight - 1
            tailPoints.append((pX,pY))
        return tailPoints



    def paintEvent(self, painter, width, height):
        painter.setBrush(QBrush(QColor(255,0,0), Qt.SolidPattern))
        if self.alive == False:
            painter.setBrush(QBrush(QColor(255,0,255), Qt.SolidPattern))
        painter.drawRect(self.headX * width, self.headY *height, width, height)
        pX = self.headX
        pY = self.headY
        for pX, pY in self.getTailPoints():
            painter.drawRect(pX * width, pY * height, width, height)
        painter.setBrush(QBrush(QColor(0,0,255), Qt.SolidPattern))
        if self.alive == True:
            painter.drawRect(self.fruitX * width, self.fruitY *height, width, height)

class SnakeViewer(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self)
        print "Starting Viewer"

        self.timerTime = 100

        self.debug = False

        self.fourcc = cv2.VideoWriter_fourcc(*"MP42")
        self.out = cv2.VideoWriter("output.avi", self.fourcc, 20, (400,400))

        self.numCols = 30
        self.numRows = 30
        self.timer = QTimer()
        #self.timer.timeout.connect(self.update)
        self.timer.start(self.timerTime)
        self.setGeometry(400,400, 1000, 900)
        self.setWindowTitle("Test")
        self.show()

        self.snakes = []
        self.snake = None
        self.winner = None
        if self.debug == True:
            self.snake = Snake(self.numCols, self.numRows)
            self.timer.timeout.connect(self.showWinner)
            self.timer.start(self.timerTime)
        else:

            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             "config-feedforward")

            population = neat.Population(config)
            population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            population.add_reporter(stats)
            population.add_reporter(neat.Checkpointer(5))

            self.winner = population.run(self.update, 3000)
            self.network = neat.nn.FeedForwardNetwork.create(self.winner, config)

            self.timer.timeout.connect(self.showWinner)
            self.timer.start(self.timerTime)

    def screenShot(self):
        p = QPixmap.grabWindow(self.winId())
        #p.save("Test.jpg", "jpg")
        img = p.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        height = img.height()
        width = img.width()
        s = img.bits().asstring(img.width() * img.height() * 4)
        newImg = np.fromstring(s, dtype=np.uint8).reshape((height, width, 4))
        #newImg = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.out.write(newImg)


    def showWinner(self):
        #self.numCols = 30
        #self.numRows = 30
        if self.snake == None or self.snake.alive == False:
            self.snake = Snake(self.numCols, self.numRows)
        i1 = self.snake.headX/ float(100)
        i2 = self.snake.headY / float(100)
        i3 = self.snake.fruitX / float(100)
        i4 = self.snake.fruitY / float(100)
        i5 = self.snake.vel/ float(10)
        i1 = interp(self.snake.headX- self.snake.fruitX, [-self.numCols, self.numCols], [-1,1])
        i2 = interp(self.snake.headY- self.snake.fruitY, [-self.numRows, self.numRows], [-1,1])
        print i1, i2, i3, i4, i5
        if self.debug == False:
            #output = self.network.activate([i1, i2, i3, i4, i5])
            output = self.network.activate([self.snake.headX - self.snake.fruitX, self.snake.headY-self.snake.fruitY])
            #output = snake.net.activate([i1, i2])[0]
            output = output.index(max(output))
            self.snake.setVel(output)
            #output = max(min(output, 3), 0)
            #self.snake.vel = int(output)
        self.snake.update()
        self.repaint()
        self.screenShot()
        self.timer.start(self.timerTime)

    def closeEvent(self, event):
        print "CLOSED"
        self.out.release()



    def update(self, geneomes, config):
        outputAmt = [0,0,0,0, 0]
        self.snakes = []
        for geneomeID, geneome in geneomes:
            self.snakes.append(Snake(self.numCols, self.numRows))
            self.snakes[len(self.snakes)-1].geneome = geneome
            self.snakes[len(self.snakes)-1].net = neat.nn.FeedForwardNetwork.create(geneome, config)

        running = True
        while running == True:
            stillAlive = False
            first = True
            highestScore = 0
            for snake in self.snakes:
                if snake.score > 5000:
                    snake.alive = False
                if snake.alive == True:
                    stillAlive = True
                i1 = snake.headX/ float(100)
                i2 = snake.headY / float(100)
                i3 = snake.fruitX / float(100)
                i4 = snake.fruitY / float(100)
                i5 = snake.vel/ float(10)
                i1 = interp(snake.headX- snake.fruitX, [-self.numCols, self.numCols], [-1,1])
                i2 = interp(snake.headY- snake.fruitY, [-self.numRows, self.numRows], [-1,1])
                #output = snake.net.activate([i1, i2, i3, i4, i5])
                #output = snake.net.activate([i1, i2])
                output = snake.net.activate([snake.headX - snake.fruitX, snake.headY-snake.fruitY])
                output = output.index(max(output))
                #output *= 10
                if snake.score >= highestScore:
                    highestScore = snake.score
                if first == True:
                    pass
                    #print output, i1, i2, i3, i4, i5
                output = max(min(output, 3), 0)
                output = int(output)
                #snake.vel = int(output)
                snake.setVel(int(output))
                if output < len(outputAmt)-1 and output >= 0:
                    outputAmt[output] += 1
                else:
                    outputAmt[4] += 1
                if snake.vel != snake.previousVel:
                    snake.previousVel = snake.vel
                snake.update()
                first = False
            if stillAlive == False:
                running = False
            #self.repaint()
            #time.sleep(.1)

        for snake in self.snakes:
            snake.geneome.fitness = snake.score
                

        '''for geneomeID, geneome in geneomes:
            self.snake = Snake(self.numCols, self.numRows)
            network = neat.nn.FeedForwardNetwork.create(geneome, config)
            while self.snake.alive:
                #i1 = self.snake.fruitX - self.snake.headX
                #i2 = self.snake.fruitY - self.snake.headY
                #i1 = interp(self.snake.fruitX - self.snake.headX, [-self.numCols, self.numCols], [0,1])
                #i2 = interp(self.snake.fruitY - self.snake.headY, [-self.numRows, self.numRows], [0,1])
                i1 = self.snake.headX/ 100
                i2 = self.snake.headY / 100
                i3 = self.snake.fruitX /100
                i4 = self.snake.fruitY /100
                i5 = self.snake.vel/10
                output = network.activate([i1, i2, i3, i4, i5])[0]
                output *= 10
                #output = max(min(output, 3), 0)
                output = int(output)
                self.snake.vel = int(output)
                if output < len(outputAmt)-1 and output >= 0:
                    outputAmt[output] += 1
                else:
                    outputAmt[4] += 1
                if self.snake.vel != self.snake.previousVel:
                    self.snake.previousVel = self.snake.vel
                self.snake.update()
                #self.repaint()

            geneome.fitness = self.snake.score'''

        print "Output amount", outputAmt




        #if self.snake.alive == False:
        #    sys.exit(0)
        #self.timer.start(self.timerTime)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            sys.exit(0)
        elif event.key() == Qt.Key_Left and self.snake.vel != 1:
            self.snake.vel = 3
        elif event.key() == Qt.Key_Right and self.snake.vel != 3:
            self.snake.vel = 1
        elif event.key() == Qt.Key_Up and self.snake.vel != 2:
            self.snake.vel = 0
        elif event.key() == Qt.Key_Down and self.snake.vel != 0:
            self.snake.vel = 2
        elif event.key() == Qt.Key_Space:
            v = self.snake.previousMoves.pop()
            self.snake.previousMoves.append(v)
            self.snake.previousMoves.append(v)

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        qp.setBrush(QBrush(QColor(0,0,0), Qt.SolidPattern))
        xDif = width / self.numCols
        yDif = height / self.numRows
        if self.debug == False and self.winner == None:
            for snake in self.snakes:
                snake.paintEvent(qp, xDif, yDif)
            #self.snake.paintEvent(qp, xDif, yDif)
        else:
            self.snake.paintEvent(qp, xDif, yDif)
        val = xDif
        for i in range(self.numCols):
            qp.drawLine(val, 0, val, height)
            val += xDif
        val = yDif
        for i in range(self.numCols):
            qp.drawLine(0, val, width, val)
            val += yDif



if __name__ == "__main__":
    qApp = QtGui.QApplication(sys.argv)
    v = SnakeViewer()
    sys.exit(qApp.exec_())

