"""Logging helpers for writing text and JSON logs to disk."""

import logging
from datetime import datetime
import json

from .disp import dispa

class FaserLog:
    """
    Write timestamped status and matrix data to a text log file using the logging module.
    """

    def __init__(self, name = "Misc", dir = "Default"):
        """
        Create a new FaserLog and initialize its underlying log file.

        The log file name is the given name suffixed with the current timestamp;
        if a directory is provided it is prepended to the file name.

        Args:
            name (str, optional): Base name for the log. Defaults to "Misc".
            dir (str, optional): Directory to place the log file in. If "Default",
                the log file is created in the current directory. Defaults to "Default".
        """
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
        self.logger.debug("Log Initiated")

    def writeToLog(self, status):
        """
        Write a status message to the log file.

        Args:
            status: Message or object to write to the log.
        """
        logging.debug(status)

    def writeMatrixToLog(self, matrix, title = "MATRIX"):
        """
        Write a formatted matrix to the log file.

        Args:
            matrix: Matrix (or other object accepted by dispa) to log.
            title (str, optional): Caption to log alongside the matrix. Defaults to "MATRIX".
        """
        logging.debug(dispa(matrix, title)[:-1])

class JSONLog:
    """
    Accumulate timestamped log entries in memory and persist them to a JSON file.
    """

    def __init__(self, logfname):
        """
        Create a new JSONLog.

        Args:
            logfname (str): Path of the JSON file to read from and write to.
        """
        self.enum = 0
        self.log = {}
        self.logfname = logfname

    def openLog(self):
        """
        Load the log from its JSON file, creating a new (empty) log file if none exists.
        """
        try:
            with open (self.logfname) as filet:
                self.log = json.load(filet)
        except:
            self.saveLog()

    def saveLog(self):
        """
        Save the current log entries, along with the total item count, to the JSON file.
        """
        self.log['num_items'] = self.enum
        with open (self.logfname, 'w') as outfile:
            json.dump(self.log, outfile)

    def writeToLog(self, item, openClose = False):
        """
        Append a new timestamped entry to the log.

        Args:
            item: Data to store for this log entry.
            openClose (bool, optional): If True, load the log from disk before
                writing and save it back to disk afterward. Defaults to False.
        """
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
