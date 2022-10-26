import faker
import psycopg2
from faker import Faker
from psycopg2 import Error
import random
fake=Faker()
try:
    conn = psycopg2.connect(user="postgres",
                            password="318295",
                            host="localhost",
                            port="5432",
                            database="postgres")
    print("Database connected successfully")
except:
    print("Database not connected successfully")
cur=conn.cursor()

"""Ürün tablosu oluşturma"""
cur.execute("DROP TABLE IF EXISTS product")
cur.execute("CREATE TABLE product(productıd SERIAL PRIMARY KEY,productname VARCHAR(20),price INT)")
product_name_list=['Lamba','Kulaklık','Monitör','Masaj Aleti','Ayakkabı','Diş Fırçası','Telefon Kılıfı','Şarj Aleti','Fön Makinesi','Biblo','Saç Maşası']
product_price=[30,150,950,350,450,25,50,50,200,40,300]
for i in range(len(product_name_list)):
    cur.execute(f"INSERT INTO product(productname,price) values('{product_name_list[i]}','{product_price[i]}')")
conn.commit()

"""Random ürün çekme."""
rnd_id=random.randint(1,5)
postgre_select_query=str(f"select * from product where productıd={rnd_id}")
cur.execute(postgre_select_query)
random_product=cur.fetchall()
for row in random_product:
    print("id=",row[0])
    print("ürün Adı:",row[1])
    print("Fiyat:",row[2])



"""Users tablosu oluşturma"""
username_list=[]
usermail_list=[]
for i in range(100):
    username_list.append(fake.unique.name())
    usermail_list.append(fake.unique.email())
cur.execute("DROP TABLE IF EXISTS users")
cur.execute("CREATE TABLE users(userıd SERIAL PRIMARY KEY,user_name VARCHAR(20),user_surname VARCHAR(20),user_email VARCHAR(50))")
for i in range(0,len(username_list)):
    cur.execute(f"INSERT INTO users(user_name, user_surname, user_email) VALUES('{(username_list[i].split())[0]}', '{(username_list[i].split())[1]}', '{str(usermail_list[i].split('@')[0])}@gmail.com')")

"""Email tablosu oluşturma"""
cur.execute("DROP TABLE IF EXISTS emails")
cur.execute("CREATE TABLE emails(emails PRIMARY KEY VARCHAR(50) NOT NULL,mail_sended BOOLEAN NOT NULL)")
conn.commit()






conn.commit()



