import time
import configparser
import os
import ftplib
import datetime
from classAI import AI
from config import *


class Main:
    def __init__(self):
        

        self.config_ini = self.read_ini('config.ini')
        #self.ftp = ftplib.FTP(self.config_ini['FTP']['server'], self.config_ini['FTP']['username'], self.config_ini['FTP']['password'])

        self.ai = AI(self.config_ini)
        self.ai.start()
        

        #self.loop()

    def read_ini(self,file_path):
        config_ini = {'CAMERA':{}, 'FTP':{}}
        config = configparser.ConfigParser()
        config.read(file_path)
        for section in config.sections():
            # print(section)
            if section == 'CAMERA':
                for key in config[section]:
                    config_ini['CAMERA'][key] = (config[section][key])

            if section == 'FTP':
                for key in config[section]:
                    config_ini['FTP'][key] = (config[section][key])
        return config_ini
    
    def directory_exists(self,dir):
        filelist = []
        self.ftp.retrlines('LIST', filelist.append)

        for f in filelist:
             if f.split()[-1] == dir and f.split()[-2].upper().startswith('<DIR>'):
                return True
        return False

    def chdir(self, dir):
        print(dir)
        if not self.directory_exists(dir):
            listfolder = self.ftp.nlst()

            if dir not in listfolder:
                self.ftp.mkd('./' + dir)
            self.ftp.cwd('./' + dir)
            print('create folder')
        print('exist folder')

    def loop(self):
        print("ftp triggered")
        cnt_uploaded = 0
        while True:
            fs = os.listdir('./temp')
            if fs.__len__() > 0:
                png_files = [f for f in fs if os.path.splitext(f)[1].lower() == '.png']
                if png_files.__len__() > 0:
                    for png in png_files:

                        todate = datetime.datetime.now().strftime('%Y_%m_%d')
                        upload_folder_path = './' + todate

                        self.chdir(todate)

                        if upload_folder_path == "":
                            upload_path = f"STOR {png}"
                        else:
                            upload_path = f"STOR {upload_folder_path}/{png}"

                        print(upload_path)
                        img_file = open('./temp/' + png, 'rb')  # file to send
                        self.ftp.storbinary(upload_path, img_file)  # send the file
                        img_file.close()
                        os.remove(os.path.join('./temp', png))
                        cnt_uploaded += 1
                        print('uploaded : ', cnt_uploaded)

            time.sleep(2)



    pass # end of main class


if __name__ == '__main__':

    mainobj = Main()