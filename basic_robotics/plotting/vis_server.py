from flask import Flask, render_template
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for, send_from_directory, send_file

import numpy
from stl import mesh
from io import BytesIO
import time
from os.path import exists
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)

contenttypes = {
    "js":"application/javascript",
    "html":"text/html",
    "jpg":"image/jpeg",
    "dae":"model/vnd.collada+xml",
    "ico":"image/x-icon",
    "css":"text/css",
    "svg":"image/svg+xml",
    "glb":"application/octet-stream"
}

ALL_DATA = {}
CHANGED = {}

@app.route("/api/json", methods=['POST', 'GET', 'PUT', 'OPTIONS'])
def json_handler():
    global ALL_DATA
    PruneExpired()
    if request.method == 'POST':
        jdict = request.json
        if jdict == None:
            return make_response(jsonify(message = 'No JSON Provided'), 400)
        if "Key" in jdict:
            keyval = jdict["Key"]
            if keyval in ALL_DATA:
                ALL_DATA[keyval][0] = jdict
            else:
                ALL_DATA[keyval] = [jdict]
        elif "Keys" in jdict:
            if keyval in ALL_DATA:
                for key in jdict["Keys"]:
                    ALL_DATA[key][0] = jdict[key]
            else:
                for key in jdict["Keys"]:
                    ALL_DATA[key] = [jdict["Keys"][key]]
        else:
            return make_response(jsonify(message = 'No Key(s) Specified'), 400)
        return make_response(jsonify(message = 'Success'), 200)
    elif request.method == 'GET':
        params = request.args
        if "Category" in params:
            cat_specific = {}
            for key in ALL_DATA:
                if "Category" not in ALL_DATA[key][0] or ("Category" in ALL_DATA[key][0] and ALL_DATA[key][0]["Category"] == params["Category"]):
                    cat_specific[key] = ALL_DATA[key]
        else:
            cat_specific = ALL_DATA

        if "Complete" in params:
            return make_response(jsonify(cat_specific, message='Success'), 200)
        elif "Latest" in params:
            latest_dict = {}
            for key in cat_specific:
                latest_dict[key] = cat_specific[key][-1]
            return make_response(jsonify(latest_dict), 200)
        elif "Key" in params:
            keys = params["Key"].split(',')
            if "Index" in params:
                index = int(params["Index"])
                response_dict = {}
                for key in keys:
                    if key in cat_specific:
                        response_dict[key] = cat_specific[key][index]
                return make_response(jsonify(response_dict, message='Success'), 200)
            elif "GetRange" in params and "ValueRange" in params:
                varname = params["GetRange"]
                valrange = [int(x) for x in params["ValueRange"].split(",")]
                response_dict = {}
                #Get Interoplated range TODo
            else:
                response_dict = {}
                for key in keys:
                    if key in cat_specific:
                        response_dict[key] = cat_specific[key]
                return make_response(jsonify(response_dict, message='Success'), 200)

    elif request.method=='PUT':
        jdict = request.json
        if "Key" in jdict:
            keyval = jdict["Key"]
            ALL_DATA[keyval] = [jdict]
            return make_response(jsonify(message = 'Data Overridden'), 200)
        elif "Keys" in jdict:
            for keyval in jdict["Keys"]:
                ALL_DATA[keyval] = [jdict["Keys"][keyval]]

            return make_response(jsonify(message = 'Keys Updated'), 200)
        elif "DeleteAll" in jdict:
            ALL_DATA = {}
            return make_response(jsonify(message = 'All Data Deleted'), 200)
        elif "DeleteKey" in jdict:
            del ALL_DATA[jdict["DeleteKey"]]
            return make_response(jsonify(message = 'Data Deleted'), 200)
        elif "DeleteKeys" in jdict:
            for kval in jdict["DeleteKeys"]:
                del ALL_DATA[jdict["DeleteKey"][kval]]
                return make_response(jsonify(message = 'Data Deleted'), 200)
        elif "DeleteCategory" in jdict:
            for trykey in ALL_DATA:
                if "Category" in ALL_DATA[trykey][0]:
                    if ALL_DATA[trykey][0]["Category"] == jdict["DeleteCategory"]:
                        del ALL_DATA[trykey]
            return make_response(jsonify(message = 'Data Deleted'), 200)
        else:
            return make_response(jsonify(message = 'No Key'), 400)



@app.route("/", methods=['GET'])
def index():
    if exists('templates/frame.html'):
        return render_template('frame.html', title='VisualizerIndex')
    else:
        return make_response(jsonify(message = 'File Not Found'), 404)

@app.route("/<path:file_path>", methods=['GET'])
def get_file(file_path):
    if file_path is None or file_path == '':
        return render_template('frame.html', title='Visualizer')
    else:
        print('Looking for:' + file_path)
        if exists(file_path):
            print('Found:' + file_path)
            response = send_file(file_path)
            #response = make_response(jsonify(message = data), 200)
            extension = file_path.split('.')
            if extension[1] in contenttypes:
                response.headers["Content-Type"] = contenttypes[extension[1]]
            else:
                response.headers["Content-Type"] = "application/octet-stream"
            return response
        else:
            print('Not Found: ' + file_path)
            return make_response(jsonify(message = 'File Not Found'), 404)

def PruneExpired():
    for key in ALL_DATA:
        if len(ALL_DATA[key]) == 0:
            continue
        if "TimeToLive" in ALL_DATA[key][0] and "UnixTime" in ALL_DATA[key][0]:
            for i in range(len(ALL_DATA[key])):
                if ALL_DATA[key][i]["UnixTime"] + ALL_DATA[key][i]["TimeToLive"] < time.time():
                    keepstart = i+1
            if keepstart > len(ALL_DATA[key]):
                del ALL_DATA[key]
            else:
                ALL_DATA[key] = ALL_DATA[key][keepstart:]

if __name__ == "__main__":
    import os,sys
    app.run()
