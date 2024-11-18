import zmq
import zmq.asyncio
import asyncio

class RouterBroker:
    def __init__(self, port):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")

    async def relay_messages(self):
        while True:
            # Receive a message from a DEALER
            identity, message = await self.socket.recv_multipart()
            print(f"Router received: {message} from {identity}")
            # Relay the message to all connected DEALERs (broadcast)
            await self.socket.send_multipart([identity, message])