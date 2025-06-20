#!/bin/bash

# =============================================================================
# RAG Chatbot Deployment Script
# =============================================================================
# Production-ready deployment automation for RAG chatbot system
# Supports development and production environments

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="ragchatbot"
IMAGE_TAG="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        log_warn ".env file not found. Please create one based on the README instructions."
        return 1
    fi
    
    log_info "Prerequisites check passed!"
    return 0
}

# Function to build the application
build() {
    log_header "Building RAG Chatbot Application"
    
    local env_type=${1:-"dev"}
    local docker_file=""
    
    if [ "$env_type" = "prod" ]; then
        docker_file="docker-compose.prod.yml"
    else
        docker_file="docker-compose.yml"
    fi
    
    log_info "Building images for $env_type environment..."
    docker-compose -f "$docker_file" build --no-cache
    
    log_info "Build completed successfully!"
}

# Function to start services
up() {
    log_header "Starting RAG Chatbot Services"
    
    local env_type=${1:-"dev"}
    local docker_file=""
    
    if [ "$env_type" = "prod" ]; then
        docker_file="docker-compose.prod.yml"
    else
        docker_file="docker-compose.yml"
    fi
    
    log_info "Starting services for $env_type environment..."
    
    # Stop existing containers first
    docker-compose -f "$docker_file" down -v || true
    
    # Remove old images if they exist
    log_info "Cleaning up old images..."
    docker images | grep "$PROJECT_NAME" | awk '{print $3}' | xargs -r docker rmi -f || true
    
    # Build and start services
    docker-compose -f "$docker_file" up -d
    
    log_info "Services started successfully!"
    
    # Show service information
    show_services_info "$env_type"
}

# Function to stop services
down() {
    log_header "Stopping RAG Chatbot Services"
    
    local env_type=${1:-"dev"}
    local docker_file=""
    
    if [ "$env_type" = "prod" ]; then
        docker_file="docker-compose.prod.yml"
    else
        docker_file="docker-compose.yml"
    fi
    
    log_info "Stopping services for $env_type environment..."
    docker-compose -f "$docker_file" down -v
    
    log_info "Services stopped successfully!"
}

# Function to show service status
status() {
    log_header "RAG Chatbot Service Status"
    
    local env_type=${1:-"dev"}
    local docker_file=""
    
    if [ "$env_type" = "prod" ]; then
        docker_file="docker-compose.prod.yml"
    else
        docker_file="docker-compose.yml"
    fi
    
    docker-compose -f "$docker_file" ps
}

# Function to show logs
logs() {
    local env_type=${1:-"dev"}
    local service=${2:-""}
    local docker_file=""
    
    if [ "$env_type" = "prod" ]; then
        docker_file="docker-compose.prod.yml"
    else
        docker_file="docker-compose.yml"
    fi
    
    if [ -n "$service" ]; then
        log_info "Showing logs for service: $service"
        docker-compose -f "$docker_file" logs -f "$service"
    else
        log_info "Showing logs for all services"
        docker-compose -f "$docker_file" logs -f
    fi
}

# Function to restart services
restart() {
    log_header "Restarting RAG Chatbot Services"
    
    local env_type=${1:-"dev"}
    
    down "$env_type"
    sleep 2
    up "$env_type"
}

# Function to clean up everything
clean() {
    log_header "Cleaning Up RAG Chatbot Environment"
    
    log_warn "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # Stop all services
        docker-compose -f docker-compose.yml down -v || true
        docker-compose -f docker-compose.prod.yml down -v || true
        
        # Remove project images
        log_info "Removing project images..."
        docker images | grep "$PROJECT_NAME" | awk '{print $3}' | xargs -r docker rmi -f || true
        
        # Remove dangling images
        log_info "Removing dangling images..."
        docker image prune -f
        
        # Remove unused volumes
        log_info "Removing unused volumes..."
        docker volume prune -f
        
        log_info "Cleanup completed!"
    else
        log_info "Cleanup cancelled."
    fi
}

# Function to show service information
show_services_info() {
    local env_type=${1:-"dev"}
    
    echo ""
    log_header "Service Information"
    
    if [ "$env_type" = "prod" ]; then
        log_info "Production Environment Services:"
        log_info "- FastAPI + Gradio: http://localhost:8081"
        log_info "- API Documentation: http://localhost:8081/docs"
        log_info "- Health Check: http://localhost:8081/health"
        log_info "- Qdrant Dashboard: http://localhost:6333/dashboard"
        log_info "- PostgreSQL: localhost:5432"
    else
        log_info "Development Environment Services:"
        log_info "- FastAPI: http://localhost:8081"
        log_info "- Gradio Interface: http://localhost:7860"
        log_info "- API Documentation: http://localhost:8081/docs"
        log_info "- Health Check: http://localhost:8081/health"
        log_info "- Qdrant Dashboard: http://localhost:6333/dashboard"
        log_info "- PostgreSQL: localhost:5432"
    fi
    
    echo ""
    log_info "To view logs: ./deploy.sh logs [$env_type] [service_name]"
    log_info "To check status: ./deploy.sh status [$env_type]"
    echo ""
}

# Function to show usage
usage() {
    echo "RAG Chatbot Deployment Script"
    echo ""
    echo "Usage: $0 <command> [environment] [options]"
    echo ""
    echo "Commands:"
    echo "  build [dev|prod]     Build application images"
    echo "  up [dev|prod]        Start all services (default: dev)"
    echo "  down [dev|prod]      Stop all services"
    echo "  restart [dev|prod]   Restart all services"
    echo "  status [dev|prod]    Show service status"
    echo "  logs [dev|prod] [service]  Show logs (optionally for specific service)"
    echo "  clean                Clean up all containers, images, and volumes"
    echo "  help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 up                Start development environment"
    echo "  $0 up prod           Start production environment"
    echo "  $0 logs prod chatbot_app  Show logs for chatbot_app in production"
    echo "  $0 clean             Clean up everything"
}

# Main script logic
main() {
    local command=${1:-"help"}
    local env_type=${2:-"dev"}
    local extra_param=$3
    
    # Validate environment type
    if [[ "$env_type" != "dev" && "$env_type" != "prod" ]]; then
        log_error "Invalid environment type: $env_type. Use 'dev' or 'prod'."
        exit 1
    fi
    
    case $command in
        "build")
            check_prerequisites || exit 1
            build "$env_type"
            ;;
        "up")
            check_prerequisites || exit 1
            up "$env_type"
            ;;
        "down")
            down "$env_type"
            ;;
        "restart")
            check_prerequisites || exit 1
            restart "$env_type"
            ;;
        "status")
            status "$env_type"
            ;;
        "logs")
            logs "$env_type" "$extra_param"
            ;;
        "clean")
            clean
            ;;
        "help"|"--help"|"-h")
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 