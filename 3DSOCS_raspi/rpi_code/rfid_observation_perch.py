import time
import serial
import datetime as dt
import RPi.GPIO as GPIO
import os


#run python -m serial.tools.list_ports in terminal to check serial port name
name="rfid1"
def mof_read(ser):  
    print("enter mof")
    ser.write(b"MOF\r")
    print("MOF")
    time.sleep(2.5)
    while True:
        if ser.inWaiting() > 0:
            data = ser.read_until("\r".encode())[0:-1]
            print(data)
            return

def arrival_check(ser, tag_present,file_name):
    # global scrounge_count
    while tag_present==0:
        if ser.inWaiting() > 0:
            id_tag = ser.read_until("\r".encode())[0:-1]
            id_tag = id_tag.decode("latin-1")
            #print(id_tag)
            #print (len(id_tag))
            if len(id_tag)==10:
                CurrentTime = dt.datetime.now()
                time_stamp = CurrentTime.strftime('%Y-%m-%d %H:%M:%S:%f').split()
                
                Epoch_Time = int(CurrentTime.timestamp()*1000000)
                write_csv("{},{},{},{},{},{}".format(id_tag,"Arrive",time_stamp[0],time_stamp[1],Epoch_Time,name),file_name)
                tag_present = 1
                print("{} --- {} arrived".format(time_stamp[1], id_tag[-10:]))
            else:
                print("noise")
                    
    return tag_present, id_tag

def depart(ser, tag_present, id_tag, file_name):
    
    tolerance_limit = 0
    
    while tag_present==1:
        ser.write("RSD\r".encode())
        
        time.sleep(.2)
        if ser.inWaiting() > 0:
            data = ser.read_until("\r".encode())[0:-1]
            data = data.decode("latin-1")
            #print(data)
            if (data == "?1"): #nothing detected, bird left
                tolerance_limit +=1
                if tolerance_limit > 10:
                    CurrentTime = dt.datetime.now()
                    time_stamp = CurrentTime.strftime('%Y-%m-%d %H:%M:%S:%f').split()
                    Epoch_Time = int(CurrentTime.timestamp()*1000000)
                    print("{} --- {} departed".format(time_stamp[1], id_tag))
                    write_csv("{},{},{},{},{},{}".format(id_tag,"Depart",time_stamp[0],time_stamp[1],Epoch_Time,name),file_name)
                    tag_present=0

            
            elif(data[-10:] != id_tag and id_tag[-4:] not in data):
                print("displacement")
                CurrentTime = dt.datetime.now()
                time_stamp = CurrentTime.strftime('%Y-%m-%d %H:%M:%S:%f').split()
                Epoch_Time = int(CurrentTime.timestamp()*1000000)
                id_tag = data
                write_csv("{},{},{},{},{},{}".format(id_tag,"Displace",time_stamp[0],time_stamp[1],Epoch_Time, name),file_name)
                print("{} --- {} displacement".format(time_stamp[1], id_tag))
            else:
                tolerance_limit=0
            
    return tag_present


def write_csv(to_write_vec,file_name):
    with open(file_name, "a") as savefile:
        savefile.write(to_write_vec+"\n")
        

if __name__ == "__main__":
    ser = serial.Serial('/dev/ttyAMA0', baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )

    mof_read(ser)

    #prepare csv
    time_stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
    date_stamp = dt.datetime.now().strftime('%Y-%m-%d')
    file_name = "Data/{}_RFID_{}.csv".format(date_stamp,name)
    if not os.path.exists(file_name):
            ##If path exists, no need to write column names
        write_csv("{},{},{},{},{},{}\n".format("id","status","date","time","epoch_time","location"), file_name)

    global tag_present
    tag_present = 0
    while True:
        if tag_present==0:
            tag_present, id_tag = arrival_check(ser, tag_present,file_name)
        elif tag_present==1:
            tag_present = depart(ser,tag_present,id_tag,file_name)

