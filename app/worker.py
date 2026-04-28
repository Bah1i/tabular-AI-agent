from redis import Redis
from rq import Worker, Queue
from app.core.config import settings
def main():
    conn = Redis.from_url(settings.redis_url); queue = Queue(settings.queue_name, connection=conn); Worker([queue], connection=conn).work()
if __name__ == '__main__': main()
