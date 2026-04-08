#!/bin/bash
# EasyICU Webapp 启动脚本 - 带自动重启和健康检查

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAPP_DIR="$(dirname "$SCRIPT_DIR")/src/easyicu/webapp"
LOG_FILE="/tmp/easyicu_webapp.log"
PID_FILE="/tmp/easyicu_webapp.pid"
PORT="${EASYICU_PORT:-8501}"
MAX_RETRIES=5
HEALTH_CHECK_INTERVAL=30

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        pgrep -f "streamlit run.*app.py.*$PORT" | head -1
    fi
}

health_check() {
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/_stcore/health" 2>/dev/null
}

start_webapp() {
    log "${GREEN}🚀 启动 EasyICU Webapp...${NC}"
    
    # 检查是否已运行
    if check_port; then
        log "${YELLOW}⚠️  端口 $PORT 已被占用${NC}"
        local existing_pid=$(get_pid)
        if [ -n "$existing_pid" ]; then
            log "现有进程 PID: $existing_pid"
        fi
        return 1
    fi
    
    # 启动 Streamlit
    cd "$WEBAPP_DIR" || exit 1
    
    nohup streamlit run app.py \
        --server.port=$PORT \
        --server.headless=true \
        --server.runOnSave=false \
        --server.fileWatcherType=none \
        --browser.gatherUsageStats=false \
        >> "$LOG_FILE" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # 等待启动
    log "等待服务启动..."
    for i in {1..10}; do
        sleep 1
        if check_port; then
            log "${GREEN}✅ Webapp 已启动${NC}"
            log "   📍 访问地址: http://localhost:$PORT"
            log "   📋 日志文件: $LOG_FILE"
            log "   🔢 进程 PID: $pid"
            return 0
        fi
    done
    
    log "${RED}❌ 启动超时${NC}"
    return 1
}

stop_webapp() {
    log "${YELLOW}🛑 停止 EasyICU Webapp...${NC}"
    
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        kill $pid 2>/dev/null
        sleep 2
        # 强制杀死
        if ps -p $pid > /dev/null 2>&1; then
            kill -9 $pid 2>/dev/null
        fi
        log "已停止进程 $pid"
    fi
    
    # 清理可能残留的进程
    pkill -f "streamlit run.*app.py.*$PORT" 2>/dev/null
    
    rm -f "$PID_FILE"
    log "${GREEN}✅ 已停止${NC}"
}

restart_webapp() {
    stop_webapp
    sleep 1
    start_webapp
}

status_webapp() {
    echo "========================================"
    echo "   EasyICU Webapp 状态"
    echo "========================================"
    
    if check_port; then
        local pid=$(get_pid)
        local health=$(health_check)
        echo -e "状态: ${GREEN}运行中${NC}"
        echo "端口: $PORT"
        echo "PID:  $pid"
        echo "健康: $health"
        
        if [ "$health" = "200" ]; then
            echo -e "访问: ${GREEN}http://localhost:$PORT${NC}"
        else
            echo -e "访问: ${YELLOW}可能需要刷新${NC}"
        fi
    else
        echo -e "状态: ${RED}未运行${NC}"
    fi
    echo "========================================"
}

# 守护模式 - 持续监控并自动重启
daemon_mode() {
    log "${GREEN}🔄 启动守护模式...${NC}"
    log "健康检查间隔: ${HEALTH_CHECK_INTERVAL}秒"
    
    local retry_count=0
    
    # 确保 webapp 运行
    if ! check_port; then
        start_webapp
    fi
    
    while true; do
        sleep $HEALTH_CHECK_INTERVAL
        
        if ! check_port; then
            log "${YELLOW}⚠️  检测到服务停止，尝试重启...${NC}"
            retry_count=$((retry_count + 1))
            
            if [ $retry_count -ge $MAX_RETRIES ]; then
                log "${RED}❌ 重启次数过多 ($MAX_RETRIES)，退出守护模式${NC}"
                exit 1
            fi
            
            start_webapp
            if check_port; then
                retry_count=0
                log "${GREEN}✅ 重启成功${NC}"
            fi
        else
            # 健康检查
            local health=$(health_check)
            if [ "$health" != "200" ]; then
                log "${YELLOW}⚠️  健康检查失败 (HTTP $health)，尝试重启...${NC}"
                restart_webapp
            fi
        fi
    done
}

# 显示帮助
show_help() {
    echo "EasyICU Webapp 管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start    启动 webapp"
    echo "  stop     停止 webapp"
    echo "  restart  重启 webapp"
    echo "  status   查看状态"
    echo "  daemon   守护模式（自动重启）"
    echo "  log      查看日志"
    echo "  help     显示帮助"
    echo ""
    echo "环境变量:"
    echo "  EASYICU_PORT  指定端口（默认 8501）"
}

# 主入口
case "$1" in
    start)
        start_webapp
        ;;
    stop)
        stop_webapp
        ;;
    restart)
        restart_webapp
        ;;
    status)
        status_webapp
        ;;
    daemon)
        daemon_mode
        ;;
    log)
        tail -f "$LOG_FILE"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
