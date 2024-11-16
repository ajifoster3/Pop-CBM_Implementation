import zmq
import zmq.asyncio
import asyncio

# Async Subscriber Class
class AsyncSubscriber:
    def __init__(self, address, port, topics=None):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{address}:{port}")
        print(f"Async Subscriber connected to tcp://{address}:{port}")

        # Subscribe to specific topics or all if topics=None
        if topics:
            for topic in topics:
                self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
                print(f"Subscribed to topic: {topic}")
        else:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics
            print("Subscribed to all topics")

    async def receive(self):
        """Receive a message from the publisher asynchronously."""
        message = await self.socket.recv_string()
        print(f"Received: {message}")
        # Parse the message
        parts = message.split(" ", 2)
        if len(parts) == 3:
            topic, sender_id, content = parts
            return topic, sender_id, content
        else:
            return None, None, message
