#!/bin/bash

# Configuration
APP_MODULE="infer:app"
HOST="0.0.0.0"
PORT=8001
PID_FILE="logs/app.pid"

start() {
    echo "Starting Fun-ASR Fast API Service..."
    if [ -f "$PID_FILE" ]; then
        if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Service is already running with PID $(cat $PID_FILE)."
            exit 1
        else
            echo "Removing stale PID file $PID_FILE"
            rm -f "$PID_FILE"
        fi
    fi
    
    # 确保 logs 目录存在
    mkdir -p logs

    # 这里使用 uvicorn 启动
    uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" &
    PID=$!
    echo $PID > "$PID_FILE"
    echo "Service started with PID $PID."
    
    # 因为这是在 Docker 容器内部启动的主进程，我们需要防止退出
    # 如果是在后台运行，则 wait
    wait $PID
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping Fun-ASR Fast API Service with PID $PID..."
        kill -15 "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "Service stopped."
    else
        echo "Service is not running (PID file not found)."
    fi
}

restart() {
    stop
    sleep 2
    start
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac
