import zmq
from zmq import asyncio

class DealerAgent:
    def __init__(self, port, agent_id):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, str(agent_id).encode())
        self.socket.connect(f"tcp://localhost:{port}")

    async def send(self, message):
        await self.socket.send_string(message)
        print(f"Agent sent: {message}")

    async def receive(self):
        message = await self.socket.recv_string()
        print(f"Agent received: {message}")