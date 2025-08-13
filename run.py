#!/usr/bin/env python3
"""
Face Mask Detection System Runner
Simple script to run the application with different configurations
"""

import os
import sys
import argparse
import subprocess
import signal
import atexit
from pathlib import Path

class ApplicationRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = []
        
        # Register cleanup handler
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up running processes"""
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
    
    def run_development(self):
        """Run in development mode"""
        print("Starting Face Mask Detection System in DEVELOPMENT mode...")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'FLASK_ENV': 'development',
            'FLASK_DEBUG': 'true',
            'PYTHONPATH': str(self.project_root)
        })
        
        # Start Flask development server
        try:
            process = subprocess.Popen([
                sys.executable, 'app.py'
            ], env=env, cwd=self.project_root)
            
            self.processes.append(process)
            
            print("✓ Flask server started on http://localhost:5000")
            print("✓ API available at http://localhost:5000/api/health")
            print("✓ Press Ctrl+C to stop")
            
            process.wait()
            
        except KeyboardInterrupt:
            print("\nShutting down development server...")
        except Exception as e:
            print(f"Error starting development server: {e}")
            sys.exit(1)
    
    def run_production(self):
        """Run in production mode with Gunicorn"""
        print("Starting Face Mask Detection System in PRODUCTION mode...")
        
        # Check if gunicorn is installed
        try:
            subprocess.run(['gunicorn', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Gunicorn not found. Install with: pip install gunicorn")
            sys.exit(1)
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'FLASK_ENV': 'production',
            'FLASK_DEBUG': 'false',
            'PYTHONPATH': str(self.project_root)
        })
        
        # Start Gunicorn server
        try:
            cmd = [
                'gunicorn',
                '--bind', '0.0.0.0:5000',
                '--workers', '4',
                '--worker-class', 'sync',
                '--timeout', '120',
                '--keepalive', '5',
                '--max-requests', '1000',
                '--max-requests-jitter', '50',
                '--access-logfile', 'logs/access.log',
                '--error-logfile', 'logs/error.log',
                '--log-level', 'info',
                'app:app'
            ]
            
            process = subprocess.Popen(cmd, env=env, cwd=self.project_root)
            self.processes.append(process)
            
            print("✓ Gunicorn server started on http://0.0.0.0:5000")
            print("✓ Workers: 4")
            print("✓ Logs: logs/access.log, logs/error.log")
            print("✓ Press Ctrl+C to stop")
            
            process.wait()
            
        except KeyboardInterrupt:
            print("\nShutting down production server...")
        except Exception as e:
            print(f"Error starting production server: {e}")
            sys.exit(1)
    
    def run_docker(self):
        """Run using Docker Compose"""
        print("Starting Face Mask Detection System with DOCKER...")
        
        # Check if docker-compose is available
        try:
            subprocess.run(['docker-compose', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Docker Compose not found. Please install Docker and Docker Compose")
            sys.exit(1)
        
        try:
            # Build and start containers
            print("Building Docker images...")
            subprocess.run(['docker-compose', 'build'], cwd=self.project_root, check=True)
            
            print("Starting containers...")
            process = subprocess.Popen(['docker-compose', 'up'], cwd=self.project_root)
            self.processes.append(process)
            
            print("✓ Docker containers started")
            print("✓ Frontend: http://localhost:80")
            print("✓ API: http://localhost:5000")
            print("✓ Grafana: http://localhost:3000 (admin/admin)")
            print("✓ Press Ctrl+C to stop")
            
            process.wait()
            
        except KeyboardInterrupt:
            print("\nShutting down Docker containers...")
            subprocess.run(['docker-compose', 'down'], cwd=self.project_root)
        except Exception as e:
            print(f"Error with Docker deployment: {e}")
            sys.exit(1)
    
    def run_with_frontend(self):
        """Run backend with frontend development server"""
        print("Starting Face Mask Detection System with FRONTEND...")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'FLASK_ENV': 'development',
            'FLASK_DEBUG': 'true',
            'PYTHONPATH': str(self.project_root)
        })
        
        try:
            # Start Flask backend
            backend_process = subprocess.Popen([
                sys.executable, 'app.py'
            ], env=env, cwd=self.project_root)
            self.processes.append(backend_process)
            
            print("✓ Backend started on http://localhost:5000")
            
            # Start frontend server
            try:
                # Try live-server first
                frontend_process = subprocess.Popen([
                    'npx', 'live-server', '--port=8000', '--host=localhost',
                    '--no-browser', '--cors'
                ], cwd=self.project_root)
                self.processes.append(frontend_process)
                print("✓ Frontend started on http://localhost:8000")
                
            except FileNotFoundError:
                # Fallback to Python HTTP server
                frontend_process = subprocess.Popen([
                    sys.executable, '-m', 'http.server', '8000'
                ], cwd=self.project_root)
                self.processes.append(frontend_process)
                print("✓ Frontend started on http://localhost:8000 (Python server)")
            
            print("✓ Full stack running!")
            print("✓ Open http://localhost:8000 in your browser")
            print("✓ Press Ctrl+C to stop both servers")
            
            # Wait for processes
            for process in self.processes:
                process.wait()
                
        except KeyboardInterrupt:
            print("\nShutting down full stack...")
        except Exception as e:
            print(f"Error starting full stack: {e}")
            sys.exit(1)
    
    def run_tests(self):
        """Run test suite"""
        print("Running Face Mask Detection System tests...")
        
        # Set test environment
        env = os.environ.copy()
        env.update({
            'FLASK_ENV': 'testing',
            'DATABASE_PATH': 'data/test_detection_logs.db',
            'PYTHONPATH': str(self.project_root)
        })
        
        try:
            # Run pytest if available
            try:
                result = subprocess.run([
                    'python', '-m', 'pytest', 'tests/', '-v', '--tb=short'
                ], env=env, cwd=self.project_root)
                
                if result.returncode == 0:
                    print("✓ All tests passed!")
                else:
                    print("✗ Some tests failed!")
                    sys.exit(1)
                    
            except FileNotFoundError:
                # Fallback to basic health check
                print("pytest not found, running basic health check...")
                result = subprocess.run([
                    sys.executable, '-c', 
                    'import app; print("✓ App imports successfully")'
                ], env=env, cwd=self.project_root)
                
                if result.returncode == 0:
                    print("✓ Basic health check passed!")
                else:
                    print("✗ Basic health check failed!")
                    sys.exit(1)
                    
        except Exception as e:
            print(f"Error running tests: {e}")
            sys.exit(1)
    
    def show_status(self):
        """Show system status"""
        print("Face Mask Detection System Status")
        print("=" * 40)
        
        # Check if files exist
        files_to_check = [
            'app.py', 'requirements.txt', 'config.js', 
            'index.html', 'docker-compose.yml'
        ]
        
        for file in files_to_check:
            file_path = self.project_root / file
            status = "✓" if file_path.exists() else "✗"
            print(f"{status} {file}")
        
        # Check directories
        dirs_to_check = ['data', 'models', 'logs', 'ssl']
        
        print("\nDirectories:")
        for dir_name in dirs_to_check:
            dir_path = self.project_root / dir_name
            status = "✓" if dir_path.exists() else "✗"
            print(f"{status} {dir_name}/")
        
        # Check Python packages
        print("\nPython packages:")
        packages = ['flask', 'opencv-python', 'tensorflow', 'numpy']
        
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✓ {package}")
            except ImportError:
                print(f"✗ {package}")
        
        # Check environment file
        env_file = self.project_root / '.env'
        if env_file.exists():
            print(f"✓ Environment file: {env_file}")
        else:
            print("✗ Environment file missing (.env)")
        
        print("\nTo setup missing components, run: python setup.py")

def main():
    parser = argparse.ArgumentParser(
        description='Run Face Mask Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run.py --dev              # Development mode
  python run.py --prod             # Production mode  
  python run.py --docker           # Docker deployment
  python run.py --fullstack       # Backend + Frontend
  python run.py --test            # Run tests
  python run.py --status          # Show system status
        '''
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dev', action='store_true', 
                      help='Run in development mode')
    group.add_argument('--prod', action='store_true', 
                      help='Run in production mode with Gunicorn')
    group.add_argument('--docker', action='store_true', 
                      help='Run using Docker Compose')
    group.add_argument('--fullstack', action='store_true', 
                      help='Run backend + frontend servers')
    group.add_argument('--test', action='store_true', 
                      help='Run test suite')
    group.add_argument('--status', action='store_true', 
                      help='Show system status')
    
    args = parser.parse_args()
    
    runner = ApplicationRunner()
    
    if args.dev:
        runner.run_development()
    elif args.prod:
        runner.run_production()
    elif args.docker:
        runner.run_docker()
    elif args.fullstack:
        runner.run_with_frontend()
    elif args.test:
        runner.run_tests()
    elif args.status:
        runner.show_status()

if __name__ == '__main__':
    main()
