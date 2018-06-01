#from mysql.connector import MySQLConnection
import MySQLdb

config = {
      "host": "127.0.0.1",
      "user": "root",
      "password": "root",
      "database":"allspark"
};

#cnx = MySQLConnection(**config)  
cnx = MySQLdb.connect(**config)    
cursor = cnx.cursor()

try:
    query = "SELECT * from tb_users"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    for row in rows:
        print(row)
    
        
except Exception as e:
    cnx.rollback()
finally:
    cnx.commit()
    cursor.close()
    cnx.close()