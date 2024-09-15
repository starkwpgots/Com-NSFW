import asyncio
import websockets

async def connect_to_proxy():

    uri = "wss://redesigned-engine-wr46vx99656fgprx-8765.app.github.dev"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket proxy server.")

        # Send a message to the server
        await websocket.send("Hello from client!")

        # Receive and print messages from the server
        while True:
            response = await websocket.recv()
            print(f"Received message from server: {response}")

if __name__ == "__main__":
    asyncio.run(connect_to_proxy())
