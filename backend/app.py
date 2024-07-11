from flask import Flask
from flask_cors import CORS
from flask import request
import MLmodels as ml
from MLmodels import NeighborSampler
import SimilarityWord as sw
from database.sql import SQLCommands
import database.sql as sql
import globals
app = Flask(__name__)
CORS(app)
sql_commands = SQLCommands(globals.database, globals.user, globals.password, globals.host)
@app.route('/')
def api_works():
    return 'api works'

@app.route('/bert')
def bert():
    text=request.args['text']
    text=ml.preprocess(text)
    resalt = ml.bert(text)
    return 

@app.route('/sgd2')
def sgd2():
    text=request.args['text']
    text=ml.preprocess(text)
    return ml.SGD2(text)

@app.route('/sgd4')
def sgd4():
    text=request.args['text']
    text=ml.preprocess(text)
    return ml.SGD4(text)

@app.route('/sgd2piplines')
def sgd2piplines():
    text=request.args['text']
    text=ml.preprocess(text)
    return ml.SGD2Piplines(text)

@app.route('/sgd22')
def sgd22():
    text=request.args['text']
    text=ml.preprocess(text)
    class_name=request.args['class']
    return ml.SGD2_2(class_name, text)

@app.route('/main')
def main():
   try:
    text=request.args['text']
    text=ml.preprocess(text)
    bert = ml.bert(text)
    sgd2 = ml.SGD2(text)
    sgd2_p = ml.SGD2Piplines(text)

    sgd4 = str(ml.SGD4(text))
    bert4 = str(bert)+ str(ml.SGD2_2(bert, text))
    sgd24 = str(sgd2)+ str(ml.SGD2_2(sgd2, text))
    sgd2_p4 = str(sgd2_p)+ str(ml.SGD2_2(sgd2_p, text))
    main = [sgd4,bert4,sgd24,sgd2_p4]
    main = str(max(set(main),key =main.count))
    resalt = {'bert':bert4,'sgd4':sgd4,'sgdpipline':sgd2_p4,'sgd':sgd24,'main':main,'code':sw.get10code(text,main)}
    return {'response':True,'items':resalt}
   except:
    return {'response':False,'items':{'main':'0000','code':'000000'}}

@app.route('/onchange')
def onchange():
   try:
    text=request.args['text']
    text=ml.preprocess(text)
    sgd2 = ml.SGD2(text)
    sgd24 = str(sgd2)+ str(ml.SGD2_2(sgd2, text))
    if len(sgd24)<4:
        sgd24='0'+sgd24
    return {'response':True,'items':{'main':{'code1':sgd24[:-2],'code2':sgd24[-2:]},'code':sw.get10code(text,sgd24)}}
   except:
    return {'response':False,'items':{'main':{'code1':'00','code2':'00'},'code':'000000'}}

@app.route('/getlist')
def getlist():
     return sql.sql_select('tnvd',{})

@app.route('/setlist')
def setlist():
     name = request.args['name']
     autor = request.args['autor']
     date = request.args['date']
     code = request.args['code']
     return sql.sql_insert('tnvd',{'name':name,'autor':autor,'data':date,'code':code})

@app.route("/sign_up", methods=["POST", "GET"])
def sign_up():
    global sql_commands

    if request.method == "POST":
        sql_args = dict()
        for key in ("login", "password", "name", "surname", "patronymic"):
            try:
                sql_args[key] = request.form[key]
            except KeyError:
                return {"response": False, "error": "no error", "id": None, "description": f"no {key} found"}

        sql_commands.open_connection()

        response = sql_commands.select("users", {"login": sql_args["login"]})

        if not response["response"]:
            sql_commands.close_connection()

            return {"response": False, "error": response["error"], "id": None, "description": "error in sql request"}
        else:
            if len(response["items"]) != 0:
                sql_commands.close_connection()

                return {"response": False, "error": "no error", "id": None, "description": "login already in use"}

        rv = sql_commands.insert("users", sql_args)

        sql_commands.close_connection()

        return {"response": True, "error": "no error", "id": rv["id"], "description": "user successfully added"}

    return {"response": False, "error": "no error", "id": None, "description": "no request found"}


@app.route("/log_in", methods=["POST", "GET"])
def log_in():
    global sql_commands

    if request.method == "GET":
        sql_args = dict()
        for key in ("login", "password"):
            try:
                sql_args[key] = request.args[key]
            except KeyError:
                return {"response": False, "error": "no error", "items": None, "description": f"no {key} found"}

        sql_commands.open_connection()

        rv = sql_commands.select("users", sql_args)

        sql_commands.close_connection()

        if not rv["response"]:
            return {"response": False, "error": rv["error"], "items": None, "description": "error in sql request"}
        else:
            if len(rv["items"]) == 0:
                return {"response": False, "error": "no error", "items": None, "description": "no users found"}

        rv["items"][0].pop("password")
        rv["items"][0].pop("login")

        return {"response": True, "error": "no error", "items": rv["items"], "description": "user successfully found"}

    return {"response": False, "error": "no error", "items": None, "description": "no request found"}
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8711)