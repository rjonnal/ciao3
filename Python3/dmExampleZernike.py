#!/usr/bin/python

import sys
import os
import time
import struct
import csv

''' Add '/Lib' or '/Lib64' to path '''
if (8 * struct.calcsize("P")) == 32:
    print("Use x86 libraries.")
    from Lib.asdk import DM
else:
    print("Use x86_64 libraries.")
    from Lib64.asdk import DM


''' Start example '''
def main(args):
    print("Please enter the S/N within the following format BXXYYY (see DM backside): ")
    serialName = sys.stdin.readline().rstrip()
    
    print("Connect the mirror")
    dm = DM( serialName )
        
    print("Retrieve number of actuators")
    nbAct = int( dm.Get('NBOfActuator') )
    print( "Number of actuator for " + serialName + ": " + str(nbAct) )
    
    print("Send 0 on each actuators")
    values = [0.] * nbAct
    dm.Send( values )

    #Load matrix Zernike to command matrix
    Z2C=[]
    with open('./config/'+serialName+'-Z2C.csv', newline='') as csvfile:
         csvrows = csv.reader(csvfile, delimiter=' ')
         for row in csvrows:
             x=row[0].split(",")
             Z2C.append(x)
    for i in range(len(Z2C)):
        for j in range(len(Z2C[i])):
            Z2C[i][j]=1.*float(Z2C[i][j])

    #Apply first Zernike one by one on mirror (use only 15 modes)
    print('Send Zernike to the mirror: #XX')      
    for zern in range(15):
        dm.Send(Z2C[zern])
        print(str(zern)+" zern")
        time.sleep(1) # Wait for 1 second

    print("Reset")
    dm.Reset()
    
    print("Exit")

if __name__=="__main__":
    main(sys.argv)
