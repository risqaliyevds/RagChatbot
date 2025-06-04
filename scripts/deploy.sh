#!/bin/bash

# Chatbot Application Docker Deployment Script
# ============================================

set -e

echo "🚀 Starting Chatbot Application Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker Compose is installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    # Create data directory and subdirectories if they don't exist
    mkdir -p ./data/qdrant_storage 2>/dev/null || true
    mkdir -p ./data/postgres_data 2>/dev/null || true
    mkdir -p ./data/app_data 2>/dev/null || true
    
    # Try to set proper permissions, but don't fail if we can't
    print_status "Setting directory permissions..."
    
    # Check if we can write to the directories, if not, try with sudo
    if [ -w ./data/qdrant_storage ]; then
        chmod 755 ./data/qdrant_storage 2>/dev/null || print_warning "Could not set permissions for data/qdrant_storage"
    else
        print_warning "data/qdrant_storage directory exists but is not writable by current user"
        print_status "Attempting to fix permissions with sudo..."
        sudo chown -R $USER:$USER ./data/qdrant_storage 2>/dev/null || print_warning "Could not change ownership of data/qdrant_storage"
        chmod 755 ./data/qdrant_storage 2>/dev/null || print_warning "Could not set permissions for data/qdrant_storage"
    fi
    
    if [ -w ./data/postgres_data ]; then
        chmod 755 ./data/postgres_data 2>/dev/null || print_warning "Could not set permissions for data/postgres_data"
    else
        print_warning "data/postgres_data directory exists but is not writable by current user"
        print_status "Attempting to fix permissions with sudo..."
        sudo chown -R $USER:$USER ./data/postgres_data 2>/dev/null || print_warning "Could not change ownership of data/postgres_data"
        chmod 755 ./data/postgres_data 2>/dev/null || print_warning "Could not set permissions for data/postgres_data"
    fi
    
    if [ -w ./data/app_data ]; then
        chmod 755 ./data/app_data 2>/dev/null || print_warning "Could not set permissions for data/app_data"
    else
        print_warning "data/app_data directory exists but is not writable by current user"
        print_status "Attempting to fix permissions with sudo..."
        sudo chown -R $USER:$USER ./data/app_data 2>/dev/null || print_warning "Could not change ownership of data/app_data"
        chmod 755 ./data/app_data 2>/dev/null || print_warning "Could not set permissions for data/app_data"
    fi
    
    print_success "Directory setup completed"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    docker build -t chatbot:latest . || {
        print_error "Failed to build Docker image"
        exit 1
    }
    
    print_success "Docker image built successfully"
}

# Stop existing containers
stop_containers() {
    print_status "Stopping existing containers..."
    
    docker-compose down --remove-orphans || {
        print_warning "No existing containers to stop"
    }
    
    print_success "Existing containers stopped"
}

# Start services
start_services() {
    print_status "Starting services with Docker Compose..."
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in current directory"
        exit 1
    fi
    
    # Start services in detached mode
    docker-compose up -d || {
        print_error "Failed to start services"
        exit 1
    }
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose exec -T postgres pg_isready -U chatbot_user -d chatbot_db &> /dev/null; then
            print_success "PostgreSQL is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "PostgreSQL failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for Qdrant
    print_status "Waiting for Qdrant..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:6333/collections &> /dev/null; then
            print_success "Qdrant is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Qdrant failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for main application
    print_status "Waiting for main application..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8080/health &> /dev/null; then
            print_success "Main application is ready"
            break
        fi
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Main application failed to start within 120 seconds"
        print_warning "Check logs with: docker-compose logs chatbot_app"
        exit 1
    fi
    
    # Wait for Gradio interface
    print_status "Waiting for Gradio interface..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:7860 &> /dev/null; then
            print_success "Gradio interface is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Gradio interface failed to start within 60 seconds"
        print_warning "Check logs with: docker-compose logs gradio_app"
    fi
}

# Show service status
show_status() {
    print_status "Checking service status..."
    echo ""
    docker-compose ps
    echo ""
}

# Show access information
show_access_info() {
    echo ""
    print_success "🎉 Deployment completed successfully!"
    echo ""
    echo "📋 Service Access Information:"
    echo "=============================="
    echo ""
    echo "🌐 Main API:           http://localhost:8080"
    echo "📖 API Documentation: http://localhost:8080/docs"
    echo "❤️  Health Check:      http://localhost:8080/health"
    echo "📊 Collections Info:   http://localhost:8080/v1/collections"
    echo ""
    echo "🎨 Gradio Interface:   http://localhost:7860"
    echo ""
    echo "🗄️  Database Access:"
    echo "   PostgreSQL:        localhost:5432"
    echo "   Database:          chatbot_db"
    echo "   Username:          chatbot_user"
    echo "   Password:          chatbot_password"
    echo ""
    echo "🔍 Qdrant Vector DB:   http://localhost:6333"
    echo ""
    echo "📝 Useful Commands:"
    echo "==================="
    echo "View logs:           docker-compose logs -f"
    echo "View app logs:       docker-compose logs -f chatbot_app"
    echo "Stop services:       docker-compose down"
    echo "Restart services:    docker-compose restart"
    echo "Update application:  ./deploy.sh"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "==================="
    echo "If services fail to start, check logs and ensure ports are not in use:"
    echo "  - Port 8080 (Main API)"
    echo "  - Port 7860 (Gradio)"
    echo "  - Port 5432 (PostgreSQL)"
    echo "  - Port 6333 (Qdrant)"
    echo ""
}

# Main deployment function
main() {
    echo ""
    print_status "Starting deployment process..."
    echo ""
    
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Prepare environment
    create_directories
    
    # Build and deploy
    build_image
    stop_containers
    start_services
    
    # Wait for services
    wait_for_services
    
    # Show results
    show_status
    show_access_info
}

# Handle script arguments
case "${1:-}" in
    "build")
        print_status "Building Docker image only..."
        check_docker
        build_image
        print_success "Build completed"
        ;;
    "start")
        print_status "Starting services only..."
        check_docker
        check_docker_compose
        start_services
        wait_for_services
        show_status
        show_access_info
        ;;
    "stop")
        print_status "Stopping services..."
        check_docker_compose
        docker-compose down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting services..."
        check_docker_compose
        docker-compose restart
        wait_for_services
        show_status
        print_success "Services restarted"
        ;;
    "logs")
        check_docker_compose
        docker-compose logs -f
        ;;
    "status")
        check_docker_compose
        show_status
        ;;
    "clean")
        print_warning "This will remove all containers, images, and volumes!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Cleaning up..."
            docker-compose down -v --remove-orphans
            docker rmi chatbot:latest 2>/dev/null || true
            docker system prune -f
            print_success "Cleanup completed"
        else
            print_status "Cleanup cancelled"
        fi
        ;;
    "help"|"-h"|"--help")
        echo "Chatbot Application Deployment Script"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  (no args)  Full deployment (build + start)"
        echo "  build      Build Docker image only"
        echo "  start      Start services only"
        echo "  stop       Stop all services"
        echo "  restart    Restart all services"
        echo "  logs       Show service logs"
        echo "  status     Show service status"
        echo "  clean      Remove all containers and images"
        echo "  help       Show this help message"
        echo ""
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 