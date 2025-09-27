from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    return '交易信号监控系统运行中...'

@app.route('/health')
def health():
    return {'status': 'healthy', 'service': 'trading-monitor'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)