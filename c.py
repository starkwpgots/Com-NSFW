from flask import Flask, request, jsonify
import threading
import asyncio
import websockets

app = Flask(__name__)

# WebSocket server URL
ROBOCODERS_WS_URL = 'wss://www.robocoders.ai/ws'

# Store messages temporarily
messages = []

async def websocket_handler():
    async with websockets.connect(ROBOCODERS_WS_URL) as server_ws:
        while True:
            if messages:
                message = messages.pop(0)
                await server_ws.send(message)
                response = await server_ws.recv()
                print(f"Received message from server: {response}")

@app.route('/send', methods=['POST'])
def send_message():
    content = request.json
    message = content.get('message')
    if message:
        messages.append(message)
        return jsonify({"status": "message received"})
    return jsonify({"error": "No message provided"}), 400

if __name__ == '__main__':
    # Run the WebSocket handler in a separate thread
    loop = asyncio.get_event_loop()
    threading.Thread(target=lambda: loop.run_until_complete(websocket_handler())).start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=8080)
