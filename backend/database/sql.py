import psycopg2
from psycopg2.errors import UndefinedColumn
from psycopg2.extras import DictCursor


class SQLCommands:
    def __init__(self, database, user, password, host):
        self.connection = None
        self.cursor = None

        self.database = database
        self.user = user
        self.password = password
        self.host = host

    @classmethod
    def refactor_args_for(cls, args: dict, mode: str = "update") -> (dict, str):
        if mode == "update":
            return cls.args_for_update(args)
        elif mode == "insert":
            return cls.args_for_insert(args)
        elif mode == "select":
            return cls.args_for_select(args)
        else:
            raise Exception(f"Unknown mode: {mode}")

    @classmethod
    def args_for_update(cls, args: dict) -> dict:
        row_id = None
        if "id" in args:
            row_id = args.pop("id")

        condition = ""
        for key, value in args.items():
            condition += f"{key} = '{value}', "
        condition = condition[:-2]

        return {"condition": condition, "id": row_id}

    @classmethod
    def args_for_insert(cls, args: dict) -> dict:
        columns, values = "(", "("
        for key, value in args.items():
            columns += f" {key},"
            values += f" '{value}',"

        values = values[:-1] + ")"
        columns = columns[:-1] + ")"

        return {"columns": columns, "values": values}

    @classmethod
    def args_for_select(cls, args: (dict, None)) -> (str, None):
        if args is None or len(args) == 0:
            return None

        condition = ""
        for key, value in args.items():
            if isinstance(value, (list, tuple)):
                condition += "("

                for arg in value:
                    condition += f"{key} = "
                    condition += f"'{arg}' OR "

                condition = condition[:-4] + ") AND "
            else:
                condition += f"({key} = '{value}') AND "

        condition = condition[:-5]

        return condition

    def open_connection(self):
        self.connection = psycopg2.connect(dbname=self.database, user=self.user, password=self.password, host=self.host)
        self.cursor = self.connection.cursor(cursor_factory=DictCursor)

    def close_connection(self):
        self.connection.close()
        self.cursor.close()

    def delete(self, table_name: str, row_id: str) -> dict:
        executable_string = f"DELETE FROM {table_name} WHERE id = {row_id};"
        self.cursor.execute(executable_string)

        self.connection.commit()

        return {"response": True, "id": row_id}

    def insert(self, table_name: str, args: dict) -> dict:
        args = SQLCommands.refactor_args_for(args, mode="insert")

        executable_string = f"INSERT INTO {table_name} {args['columns']} VALUES {args['values']} RETURNING id;"
        self.cursor.execute(executable_string)

        row_id = self.cursor.fetchone()[0]

        self.connection.commit()

        return {"response": True, "id": str(row_id)}

    def update(self, table_name: str, args: dict) -> dict:
        args = SQLCommands.refactor_args_for(args, mode="update")

        if args["id"] is None:
            return {"response": False, "error": "id hasn't been given"}

        executable_string = f"UPDATE {table_name} SET {args['condition']} WHERE id = {args['id']};"
        self.cursor.execute(executable_string)

        self.connection.commit()

        return {"response": True, "error": "no error"}

    def select(self, table_name: str, args: (dict, None) = None) -> dict:
        args = SQLCommands.refactor_args_for(args, mode="select")

        def refactor_records(records):
            for i in range(len(records)):
                records[i] = dict(records[i])
            return records

        if args is None:
            self.cursor.execute(f"SELECT * FROM {table_name};")

            return {"response": True, "error": "no error", "items": refactor_records(self.cursor.fetchall())}

        try:
            self.cursor.execute(f"SELECT * FROM {table_name} WHERE {args};")

            return {"response": True, "error": "no error", "items": refactor_records(self.cursor.fetchall())}
        except UndefinedColumn as undefined_column:
            return {"response": False, "error": str(undefined_column), "items": None}
        
import psycopg2
from psycopg2.extras import DictCursor , RealDictCursor
from flask import Flask , render_template ,request
from psycopg2.errors import UndefinedColumn
import time

def get_bd():
    #host = "db"
    host = "ec2-54-170-163-224.eu-west-1.compute.amazonaws.com"
    database = "d9jjphcc1pjmcp"
    user = "axmgwvuevcoqoy"
    port = 5432
    password = "dcdb2495a47edb58f517310f5786a05d3f9bfbe626ae6d084eb43377f3600ed6"
    return host , port , database , user , password



def sql_delet(table_name,param):
    host , port , database , user, password = get_bd()
    conn = psycopg2.connect(dbname=database, user=user, 
                        password=password, host=host)
    cursor = conn.cursor(cursor_factory=DictCursor)
    delet = "DELETE FROM "+str(table_name)+" WHERE id ="+str(param['id'])
    print(cursor.execute(delet))
    conn.commit()
    conn.close()
    cursor.close()
    return {'response':True}
   

def sql_insert(table_name,param):
   host , port , database , user, password = get_bd()
   param = param_insert(param)
   conn = psycopg2.connect(dbname=database, user=user, 
                        password=password, host=host)
   cursor = conn.cursor(cursor_factory=DictCursor)
   insertstr = "INSERT INTO "+str(table_name)+" "+str(param[0])+" VALUES "+str(param[1]) + " RETURNING id"
   cursor.execute(insertstr)
   _id = cursor.fetchone()[0]
   conn.commit()
   conn.close()
   cursor.close()
   return {'response':True,'id':str(_id)}
 
def sql_update(table_name,param):
   host , port , database , user, password = get_bd()
   param = param_update(param)
   if param['id']:
    conn = psycopg2.connect(dbname=database, user=user, 
                        password=password, host=host)
    cursor = conn.cursor(cursor_factory=DictCursor)
    update = "UPDATE "+str(table_name)+" SET "+str(param['colum'])+"  WHERE id ="+str(param['id'])
    cursor.execute(update)
    conn.commit()
    conn.close()
    cursor.close()
    return {'response':True}
   else:
       return {'response':False,'items': 'error id'}

def sql_select(table_name,param):
   host , port , database , user, password = get_bd()
   param = param_select(param)
   conn = psycopg2.connect(dbname=database, user=user, 
                        password=password, host=host)
   cursor = conn.cursor(cursor_factory=RealDictCursor)
   try:
    if param:
     cursor.execute('SELECT * FROM '+str(table_name)+' WHERE '+str(param))
    else:
     cursor.execute('SELECT * FROM '+str(table_name))
    records = cursor.fetchall()
    for i in range(len(records)):
        records[i]=dict(records[i])
    conn.close()
    cursor.close()
    return {'response':True, 'items':records}
   except UndefinedColumn:
     conn.close()
     cursor.close()
     return {'response':False,'items':'error colum'}

def sql_select_srt(table_name,bd_num,key,name):
   host , port , database , user, password = get_bd()
   conn = psycopg2.connect(dbname=database, user=user, 
                        password=password, host=host)
   cursor = conn.cursor(cursor_factory=RealDictCursor)
   try:
    cursor.execute("SELECT * FROM "+str(table_name)+" WHERE "+str(key)+" LIKE '%" + str(name) +"%'")
    records = cursor.fetchall()
    for i in range(len(records)):
        records[i]=dict(records[i])
    conn.close()
    cursor.close()
    return {'response':True, 'items':records}
   except UndefinedColumn:
     conn.close()
     cursor.close()
     return {'response':False,'items':'error colum'}

def sql_select_all(table_name,param):
   rezult = []
   param = param_select(param)
   for i in range(1,4):
       host , port , database , user, password = get_bd()
       conn = psycopg2.connect(dbname=database, user=user, 
                        password=password, host=host)
       cursor = conn.cursor(cursor_factory=RealDictCursor)
       try:
          if param:
               cursor.execute('SELECT * FROM '+str(table_name)+' WHERE '+str(param))
          else:
               cursor.execute('SELECT * FROM '+str(table_name))
          records = cursor.fetchall()
          for i in range(len(records)):
               records[i]=dict(records[i])
          conn.close()
          cursor.close()
          rezult+=records
       except UndefinedColumn:
          conn.close()
          cursor.close()
          return {'response':False,'items':'error colum'}
   return  {'response':True, 'items':rezult}

def sql(response, type):
       host , port , database , user, password =get_bd()
       conn = psycopg2.connect(dbname=database, user=user,
                               password=password, host=host)
       cursor = conn.cursor(cursor_factory=DictCursor)
       cursor.execute(response)
       if type == 'select':
           records = cursor.fetchall()
           for i in range(len(records)):
               records[i] = dict(records[i])
       elif type == 'insert':
           _id = cursor.fetchone()[0]
           conn.commit()
       conn.close()
       cursor.close()
       if type == 'select':
           return {'response': True, 'items': records}
       elif type == 'insert':
           return {'response': True, 'id': str(_id)}
    
def param_update(param):
    _id=None
    keys = list(param.keys())
    colum=' '
    for key in keys:
        if key=='id':
            _id=param[key]
        else:
            colum=colum+' '+str(key)+" = '" + str(param[key])+"' ,"
    colum=colum[:-1]
    return {'colum':colum,'id':_id}


def param_insert(param):
    keys = list(param.keys())
    colum=' ( '
    values = '('
    for key in keys:
        colum=colum+' '+str(key)+','
        values=values+" '"+str(param[key])+"',"
    values=values[:-1]
    values+=')'
    colum=colum[:-1]
    colum+=')'
    return colum , values
    

def param_select(param):
    condition =""
    keys = list(param.keys())
    for key in keys:
        param[key]=str(param[key])
        if ('[' and ']' in param[key]) or type(param[key])==list:
         param[key]=eval(param[key])
         condition= condition +'(' 
         for par in param[key]:
             condition=condition + str(key)+ ' = '
             condition=condition + "'"+str(par)+"'"+ ' OR '
         condition=condition[:-3]
         condition+=') AND '
         
        else:
            condition=condition+'( ' + str(key)+ ' = ' + "'" + str(param[key]) + "'" + ' ) AND '
    condition=condition[:-4]
    return condition


def get_bd_local():
    host = '194.67.91.225'
    database = "miriteam"
    user = "postgres"
    port = 5432
    password = "GF@kkjj!hdaskdh666879@gghs@@@sadadGGhac9osAlsfdf;aswmfsoHJWGHDP@@!jsl"
    return host , port , database , user , password

def sql_select_local(table_name,param):
   host , port , database , user, password = get_bd_local()
   param = param_select(param)
   conn = psycopg2.connect(dbname=database, user=user, 
                        password=password, host=host)
   cursor = conn.cursor(cursor_factory=RealDictCursor)
   try:
    if param:
     cursor.execute('SELECT * FROM '+str(table_name)+' WHERE '+str(param))
    else:
     cursor.execute('SELECT * FROM '+str(table_name))
    records = cursor.fetchall()
    for i in range(len(records)):
        records[i]=dict(records[i])
    conn.close()
    cursor.close()
    return {'response':True, 'items':records}
   except UndefinedColumn:
     conn.close()
     cursor.close()
     return {'response':False,'items':'error colum'}