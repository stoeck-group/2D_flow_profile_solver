#!/usr/bin/env python

import sys
import fileinput

def createCoor(filename):

    file = open(filename,"r")
    text = file.read()
    data = text.splitlines()


    newfile = open('newCoord.txt','w')
    newfile.write("//Points\n")
    newfile.write(str(len(data)))
    newfile.write("\n(\n")
    for i in range(len(data)):
    	di = data[i].split(" ")
    	newfile.write("(" + di[2] + " " + di[0] + " " + di[1] + " " + ")\n")
    newfile.write(")")
    
    newfile.close()
    file.close()


def createU(filename):

    file = open(filename,"r")
    text = file.read()
    data = text.splitlines()

 
    newfile = open('newVert.txt','w')

    newfile.write("// Data on points\n")
    newfile.write(str(len(data)))
    newfile.write("\n(\n")
    for i in range(len(data)):
    	newfile.write("( " + data[i] + " 0 " + "0 " ")\n")
    newfile.write(")")

    file.close()




if __name__ == '__main__':
   
   filename = sys.argv[1]
   UorP = sys.argv[2]
   if(UorP == 'P'):
   	createCoor(filename)
   elif(UorP == 'U'):
   	createU(filename)
   


