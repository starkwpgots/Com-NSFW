import asyncio
import websockets
import json
import re

# WebSocket server URL
ROBOCODERS_WS_URL = 'wss://www.robocoders.ai/ws'

def replace_urls(text):
    # Define the regex pattern to match URLs with optional paths
    url_pattern = re.compile(r'http://([a-zA-Z0-9._-]+)(/[^\s]*)?')
    
    def replacement(match):
        base_path = match.group(1)  # Capture the base URL part after http://
        path = match.group(2) if match.group(2) else ''  # Capture the path, if present
        
        # Construct the new URL with a new base URL
        new_url = f'https://backend.sathishzus.workers.dev/{base_path}{path}'
        return new_url

    # Substitute all matched URLs with the new URL
    return url_pattern.sub(replacement, text)


async def forward_messages(source_ws, target_ws):
    async for message in source_ws:
        message = message.replace("BLACKBOXAI"," Vista Ai")
        updated_message = replace_urls(message)
        updated_message = updated_message.replace(".blackbx.ai","")
        await target_ws.send(updated_message)

async def handle_client(client_ws):
    async with websockets.connect(ROBOCODERS_WS_URL) as server_ws:
        client_to_server_task = asyncio.create_task(forward_messages(client_ws, server_ws))
        server_to_client_task = asyncio.create_task(forward_messages(server_ws, client_ws))
        done, pending = await asyncio.wait(
            [client_to_server_task, server_to_client_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

async def main():
    async with websockets.serve(handle_client, 'localhost', 8765):
        print("WebSocket server is listening on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
