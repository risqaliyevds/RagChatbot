#!/usr/bin/env python3
"""
Application Startup Script
=========================

Clean entry point for the RAG chatbot application with PostgreSQL and Qdrant.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met"""
    # Configure logging first
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Checking prerequisites...")
    
    # Check if required directories exist
    required_dirs = ['documents', 'logs']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_name}")
            dir_path.mkdir(exist_ok=True)
    
    logger.info("Prerequisites check completed")


def run_fastapi_server():
    """Run the FastAPI server"""
    import uvicorn
    from app.config import load_config
    
    try:
        # Load configuration
        config = load_config()
        
        # Start the server
        uvicorn.run(
            "fastapi_application:app",
            host=config.get("host", "0.0.0.0"),
            port=config.get("port", 8081),
            reload=False,  # Set to False for production
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start FastAPI server: {e}")
        sys.exit(1)


def run_gradio_interface():
    """Run the Gradio interface"""
    try:
        from app.gradio_app import demo, test_connection
        
        # Test connection on startup
        logger.info("Testing API connection from Gradio...")
        health = test_connection()
        if "✅" not in health.get("status", ""):
            logger.warning(f"API connection test failed: {health}")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=False
        )
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chatbot Application")
    parser.add_argument(
        "--mode", 
        choices=["api", "gradio", "both"], 
        default="api",
        help="Application mode: api (FastAPI only), gradio (Gradio only), or both"
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Check database connection and initialize if needed"
    )
    
    args = parser.parse_args()
    
    # Load configuration first
    from app.config import load_config
    config = load_config()
    
    # Check prerequisites
    check_prerequisites()
    
    # Initialize database if requested
    if args.check_db:
        logger.info("Checking database connection and running initialization...")
        try:
            from app.database.postgresql_manager import init_database
            from app.database_initializer import initialize_system, check_connections
            
            # First check connections
            connections = check_connections()
            logger.info(f"Connection status: {connections}")
            
            if not all(connections.values()):
                logger.error("Some connections failed - cannot proceed")
                sys.exit(1)
            
            # Run full initialization
            import asyncio
            init_results = asyncio.run(initialize_system())
            
            if init_results['overall_success']:
                logger.info("✅ System initialization completed successfully!")
            else:
                failed_components = [k for k, v in init_results.items() if not v and k != 'overall_success']
                logger.error(f"❌ System initialization failed. Failed components: {failed_components}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Database/system check failed: {e}")
            sys.exit(1)
    
    # Run based on mode
    if args.mode == "api":
        logger.info("Starting FastAPI server...")
        run_fastapi_server()
    elif args.mode == "gradio":
        logger.info("Starting Gradio interface...")
        run_gradio_interface()
    elif args.mode == "both":
        logger.info("Starting both FastAPI and Gradio...")
        import multiprocessing
        import signal
        import time
        
        # Create processes for both services
        api_process = multiprocessing.Process(target=run_fastapi_server, name="FastAPI")
        gradio_process = multiprocessing.Process(target=run_gradio_interface, name="Gradio")
        
        def signal_handler(sig, frame):
            logger.info("\nShutting down services...")
            try:
                if api_process and api_process.is_alive():
                    api_process.terminate()
            except (AttributeError, ProcessLookupError):
                pass
            try:
                if gradio_process and gradio_process.is_alive():
                    gradio_process.terminate()
            except (AttributeError, ProcessLookupError):
                pass
            
            # Wait for processes to finish
            try:
                if api_process:
                    api_process.join(timeout=5)
            except (AttributeError, ProcessLookupError):
                pass
            try:
                if gradio_process:
                    gradio_process.join(timeout=5)
            except (AttributeError, ProcessLookupError):
                pass
            
            logger.info("Services stopped.")
            sys.exit(0)
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start both processes
            api_process.start()
            logger.info("FastAPI process started")
            
            # Give API time to start and wait for it to be ready
            startup_delay = config.get("startup_delay", 10)  # Increased from 3 to 10 seconds
            logger.info(f"Waiting {startup_delay} seconds for FastAPI to initialize...")
            time.sleep(startup_delay)
            
            # Check if FastAPI is responding before starting Gradio
            import requests
            for attempt in range(5):
                try:
                    response = requests.get("http://localhost:8081/health", timeout=5)
                    if response.status_code == 200:
                        logger.info("✅ FastAPI is ready, starting Gradio...")
                        break
                except:
                    logger.info(f"FastAPI not ready yet, attempt {attempt + 1}/5...")
                    time.sleep(3)
            else:
                logger.warning("⚠️ FastAPI may not be fully ready, but starting Gradio anyway...")
            
            gradio_process.start()
            logger.info("Gradio process started")
            
            logger.info("=" * 50)
            logger.info("Both services are running:")
            logger.info("- FastAPI: http://0.0.0.0:8081")
            logger.info("- Gradio: http://0.0.0.0:7860")
            logger.info("Press Ctrl+C to stop all services")
            logger.info("=" * 50)
            
            # Wait for processes to complete (they won't unless terminated)
            while api_process.is_alive() and gradio_process.is_alive():
                time.sleep(1)
            
            # If one process dies, kill the other
            if not api_process.is_alive():
                logger.error("FastAPI process died unexpectedly")
                if gradio_process.is_alive():
                    gradio_process.terminate()
            elif not gradio_process.is_alive():
                logger.error("Gradio process died unexpectedly")
                if api_process.is_alive():
                    api_process.terminate()
                
        except Exception as e:
            logger.error(f"Error running both services: {e}")
            if api_process and api_process.is_alive():
                api_process.terminate()
            if gradio_process and gradio_process.is_alive():
                gradio_process.terminate()
            sys.exit(1)
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        
        finally:
            # Clean shutdown
            logger.info("Shutting down processes...")
            try:
                if api_process and api_process.is_alive():
                    api_process.terminate()
            except (AttributeError, ProcessLookupError):
                pass
            try:
                if gradio_process and gradio_process.is_alive():
                    gradio_process.terminate()
            except (AttributeError, ProcessLookupError):
                pass
            
            # Wait for processes to gracefully exit
            try:
                if api_process:
                    api_process.join(timeout=config.get("process_join_timeout", 5))
            except (AttributeError, ProcessLookupError):
                pass
            try:
                if gradio_process:
                    gradio_process.join(timeout=config.get("process_join_timeout", 5))
            except (AttributeError, ProcessLookupError):
                pass


if __name__ == "__main__":
    main() 