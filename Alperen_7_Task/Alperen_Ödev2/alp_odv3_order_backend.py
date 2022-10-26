from ensurepip import bootstrap
import time
import json

from kafka import KafkaProducer
import faker
import psycopg2
from faker import Faker
from psycopg2 import Error
import random
fake=Faker()

ORDER_KAFKA_TOPIC = "order_details"
ORDER_LIMIT = 20

producer = KafkaProducer(bootstrap_servers="localhost:9092")

conn = psycopg2.connect(user="postgres",
                        password="318295",
                        host="localhost",
                        port="5432",
                        database="postgres")
cur=conn.cursor()

print("Going to be generating order after 10 seconds")
print("will generate one unique order after every 10 seconds")

try:
    cur.execute("DROP TABLE IF EXISTS orders2")
    cur.execute("CREATE TABLE orders2(orderıd INT PRIMARY KEY NOT NULL,user_ıd INT  NOT NULL,order_cost INT NOT NULL,products VARCHAR(50) )")
    print("Orders tablosu oluşturudu!!!")
except:
    print("Orders tablosu oluşturulamadı!!!")
conn.commit()

for i in range(1, ORDER_LIMIT):
    len_users_1=(cur.execute("Select COUNT(userıd) from users"))
    len_users=cur.fetchone()
    len_item_1=(cur.execute("Select COUNT(productıd) from product"))
    len_item=cur.fetchone()
    user_id_rnd=random.randint(1,int(len_users[0]))
    item_id_rnd=random.randint(1,int(len_item[0]))
    item_name_1=cur.execute(f"Select productname from product where productıd={item_id_rnd}")
    item_name=cur.fetchone()
    item_price_1=cur.execute(f"Select price from product where productıd={item_id_rnd}")
    item_price=cur.fetchone()
    conn.commit()
    data = {
        "order_id": i,
        "user_id": user_id_rnd,
        "total_cost": int(item_price[0]),
        "items": str(item_name[0])
    }
    email_2=cur.execute(f"Select user_email from users where userıd={user_id_rnd}")
    email=cur.fetchone()
    default_bool=False
    cur.execute(f"INSERT INTO orders2(orderıd,user_ıd,order_cost,products) values({i},{user_id_rnd},{data.get('total_cost')},'{str(data.get('items'))}')")
    cur.execute(f"INSERT INTO emails(emails,mail_sended) values('{email[0]}',{default_bool})")
    conn.commit()
    producer.send(
        ORDER_KAFKA_TOPIC,
        json.dumps(data).encode("utf-8")
    )

    print(f"Done sending...{i}")
    time.sleep(3)



