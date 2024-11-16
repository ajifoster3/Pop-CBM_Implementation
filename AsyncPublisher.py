import zmq
import zmq.asyncio
import asyncio

# Async Publisher Class
class AsyncPublisher:
    def __init__(self, port, sender_id):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.sender_id = sender_id
        print(f"Async Publisher (ID: {self.sender_id}) bound to port {port}")

    async def publish(self, topic, message):
        """Send a message with a sender ID under a specific topic."""
        # Ensure topic comes first for ZeroMQ filtering
        formatted_message = f"{topic} {self.sender_id} {message}"
        await self.socket.send_string(formatted_message)
        print(f"Published: {formatted_message}")
