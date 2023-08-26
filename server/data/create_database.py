import os


def create_db_if_not_exists(db_path):
    dir_path = os.path.dirname(db_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.path.exists(db_path):
        with open(db_path, 'wb') as file:
            pass
        print(f"Database {db_path} initialized")
    else:
        print(f"Database {db_path} already initialized")
