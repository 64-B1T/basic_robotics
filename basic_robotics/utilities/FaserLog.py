import logging
from datetime import datetime
import json

class FaserLog:

    def __init__(self, name = "Misc", dir = "Default"):
        now = datetime.now()
        date_time = now.strftime(" %m-%d-%Y_%H-%M-%S")
        self.name = name + date_time
        if dir != "Default":
            self.dirname = dir + "/" + self.name
        else:
            self.dirname = self.name
        logging.basicConfig(filename = self.dirname, format='%(asctime)s %(message)s', filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        logger.debug("Log Initiated")

    def writeToLog(self, status):
        logging.debug(status)
        disp("Git Test")

    def writeMatrixToLog(self, matrix, title = "MATRIX"):
        loging.debug(dispa(matrix, title)[:-1])

class JSONLog:

    def __init__(self, logfname):
        self.enum = 0
        self.log = {}
        self.logfname = logfname

    def openLog(self):
        try:
            with open (self.logfname) as filet:
                self.log = json.load(filet)
        except:
            self.saveLog()

    def saveLog(self):
        self.log['num_items'] = self.enum
        with open (self.logfname, 'w') as outfile:
            json.dump(self.log, outfile)

    def writeToLog(self, item, openClose = False):
        if openClose:
            self.openLog()
        now = datetime.now()
        date_time = now.strftime(" %m-%d-%Y_%H-%M-%S")
        self.log[str(self.enum)] = {}
        self.log[str(self.enum)]["timestamp"] = date_time
        self.log[str(self.enum)]["data"] = item
        self.enum+=1
        if openClose:
            self.saveLog()
