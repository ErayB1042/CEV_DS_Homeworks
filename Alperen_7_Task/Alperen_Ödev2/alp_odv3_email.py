from ensurepip import bootstrap
from kafka import KafkaAdminClient, KafkaConsumer
import json
import psycopg2

ORDER_KAFKA_CONFIRMED_TOPIC="order_confirmed"

consumer=KafkaConsumer(
    ORDER_KAFKA_CONFIRMED_TOPIC,
    bootstrap_servers="localhost:9092"
)
conn = psycopg2.connect(user="postgres",
                        password="318295",
                        host="localhost",
                        port="5432",
                        database="postgres")
cur=conn.cursor()

email_sent_so_far=set()
print("Email is listening")

while True:
    for message in consumer:
        consumed_message=json.loads(message.value.decode())
        customer_email=consumed_message["customer_email"]
        new_state=True
        cur.execute(f"UPDATE emails set mail_sended={new_state} where emails='{customer_email}'")
        conn.commit()
        print(f"Sending email to {customer_email}")
        email_sent_so_far.add(customer_email)
        print(f"So far emails sent to :{len(email_sent_so_far)} unique emails")