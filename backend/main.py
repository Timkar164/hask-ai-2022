from database.build_database import create_database
from database.sql import SQLCommands

from globals import *


if __name__ == '__main__':
    create_database()

    sql_commands = SQLCommands(database, user, password, host)

    sql_commands.open_connection()

    sql_commands.insert(
        "users",
        {
            "login": "alex6712",
            "password": "1AlexndR2",
            "name": "Алексей",
            "surname": "Ванюков",
            "patronymic": "Игоревич"
        }
    )

    sql_commands.close_connection()
    sql_commands.open_connection()

    sql_commands.insert(
        "users",
        {
            "login": "Den_S",
            "password": "awidhwjuawdf",
            "name": "Денис",
            "surname": "Салатов",
            "patronymic": "Витальевич"
        }
    )

    sql_commands.close_connection()

    del sql_commands
