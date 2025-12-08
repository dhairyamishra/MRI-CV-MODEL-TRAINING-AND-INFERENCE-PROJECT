# PM2 Demo Application Guide

## Overview

This guide explains how to use **PM2** (Process Manager 2) to run the SliceWise multi-task brain tumor detection demo. PM2 provides robust process management, automatic restarts, and centralized logging—making it the **recommended approach** for running the demo, especially on Windows.

## Why PM2?

### Problems with Subprocess Management
The original demo launcher used Python's `subprocess` module to manage both backend and frontend processes. This approach had several issues:

- ❌ **Windows Compatibility**: Subprocess management is unreliable on Windows
- ❌ **Process Cleanup**: Orphaned processes when parent script exits
- ❌ **No Auto-Restart**: Manual restart required if a process crashes
- ❌ **Log Management**: Difficult to access logs from background processes
- ❌ **Monitoring**: No easy way to check process status

### Benefits of PM2
- ✅ **Cross-Platform**: Works reliably on Windows, macOS, and Linux
- ✅ **Auto-Restart**: Automatically restarts processes if they crash
- ✅ **Centralized Logging**: All logs in one place with timestamps
- ✅ **Easy Monitoring**: Built-in monitoring with `pm2 monit`
- ✅ **Process Control**: Simple commands to start/stop/restart
- ✅ **Background Execution**: Processes run independently of terminal

---

## Installation

### Prerequisites
- **Node.js** and **npm** must be installed
- Check if installed: `node --version` and `npm --version`

### Install PM2 Globally

```bash
# Using npm (recommended)
npm install -g pm2

# Using yarn (alternative)
yarn global add pm2

# Verify installation
pm2 --version
```

### Windows-Specific Notes
If you encounter permission issues on Windows:

```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
npm install -g pm2
```

---

## Quick Start

### 1. Launch the Demo

```bash
# From project root
python scripts/demo/run_demo_pm2.py
```

This script will:
1. [OK] Check if PM2 is installed
2. [OK] Verify model checkpoint exists
3. [OK] Stop any existing demo processes
4. [OK] Start backend (FastAPI) and frontend (Streamlit)
5. [OK] Wait for health checks
6. [OK] Open browser to http://localhost:8501

### 2. Access the Demo

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Stop the Demo

```bash
# Stop all processes
pm2 stop all

# Or stop and remove processes
pm2 delete all

# Or stop using ecosystem config
pm2 stop configs/pm2-ecosystem/ecosystem.config.js
pm2 delete configs/pm2-ecosystem/ecosystem.config.js
```

---

## PM2 Commands Reference

### Process Management

```bash
# Start processes
pm2 start configs/pm2-ecosystem/ecosystem.config.js

# Stop all processes
pm2 stop all

# Restart all processes
pm2 restart all

# Delete all processes (stop + remove)
pm2 delete all

# Stop specific process
pm2 stop slicewise-backend
pm2 stop slicewise-frontend
```

### Monitoring & Logs

```bash
# View process status
pm2 status

# View all logs (live tail)
pm2 logs

# View backend logs only
pm2 logs slicewise-backend

# View frontend logs only
pm2 logs slicewise-frontend

# View last 100 lines
pm2 logs --lines 100

# Clear all logs
pm2 flush

# Interactive monitoring dashboard
pm2 monit
```

### Process Information

```bash
# Show detailed process info
pm2 show slicewise-backend
pm2 show slicewise-frontend

# List all processes
pm2 list

# Get process ID
pm2 pid slicewise-backend
```

### Advanced Commands

```bash
# Save process list (persist across reboots)
pm2 save

# Resurrect saved processes
pm2 resurrect

# Update PM2
npm install -g pm2@latest

# Reset PM2 (delete all processes and logs)
pm2 kill
```

---

## Ecosystem Configuration

The demo uses `ecosystem.config.js` in the `configs/pm2-ecosystem/` directory to define both processes:

```javascript
module.exports = {
  apps: [
    {
      name: 'slicewise-backend',
      script: 'uvicorn',
      args: 'app.backend.main_v2:app --host localhost --port 8000 --reload',
      interpreter: 'python',
      cwd: './',
      instances: 1,
      autorestart: true,
      max_memory_restart: '2G',
      error_file: './logs/backend-error.log',
      out_file: './logs/backend-out.log',
      log_file: './logs/backend-combined.log'
    },
    {
      name: 'slicewise-frontend',
      script: 'streamlit',
      args: 'run app/frontend/app.py --server.port 8501 --server.headless true',
      interpreter: 'python',
      cwd: './',
      instances: 1,
      autorestart: true,
      max_memory_restart: '2G',
      error_file: './logs/frontend-error.log',
      out_file: './logs/frontend-out.log',
      log_file: './logs/frontend-combined.log'
    }
  ]
};
```

### Key Features
- **Auto-restart**: Processes restart automatically if they crash
- **Memory limits**: Restart if memory exceeds 2GB
- **Separate logs**: Each process has its own log files
- **Python interpreter**: Uses your active Python environment

---

## Log Files

All logs are stored in the `logs/` directory:

```
logs/
├── backend-error.log      # Backend stderr
├── backend-out.log        # Backend stdout
├── backend-combined.log   # Backend combined
├── frontend-error.log     # Frontend stderr
├── frontend-out.log       # Frontend stdout
└── frontend-combined.log  # Frontend combined
```

### Viewing Logs

```bash
# Live tail all logs
pm2 logs

# View specific log file
cat logs/backend-error.log
cat logs/frontend-out.log

# Follow log file (like tail -f)
tail -f logs/backend-combined.log
```

---

## Troubleshooting

### PM2 Not Found

**Problem**: `pm2: command not found`

**Solution**:
```bash
# Install PM2 globally
npm install -g pm2

# Check npm global bin path
npm config get prefix

# Add to PATH if needed (Windows)
# Add C:\Users\<YourUser>\AppData\Roaming\npm to PATH

# Add to PATH if needed (macOS/Linux)
export PATH="$PATH:$(npm config get prefix)/bin"
```

### Port Already in Use

**Problem**: Port 8000 or 8501 already in use

**Solution**:
```bash
# Find and stop existing processes
pm2 list
pm2 delete all

# Or kill processes manually (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or kill processes manually (macOS/Linux)
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

### Backend Fails to Start

**Problem**: Backend process exits immediately

**Solution**:
```bash
# Check backend logs
pm2 logs slicewise-backend --lines 50

# Common issues:
# 1. Model checkpoint not found
#    → Run training: python scripts/run_full_pipeline.py --mode train-eval --training-mode quick

# 2. Missing dependencies
#    → Install: pip install -r requirements.txt

# 3. Python environment issues
#    → Activate correct environment before starting PM2
```

### Frontend Fails to Start

**Problem**: Frontend process exits immediately

**Solution**:
```bash
# Check frontend logs
pm2 logs slicewise-frontend --lines 50

# Common issues:
# 1. Streamlit not installed
#    → Install: pip install streamlit

# 2. Backend not running
#    → Start backend first: pm2 start configs/pm2-ecosystem/ecosystem.config.js --only slicewise-backend

# 3. Port conflict
#    → Change port in configs/pm2-ecosystem/ecosystem.config.js
```

### Processes Keep Restarting

**Problem**: PM2 shows processes restarting frequently

**Solution**:
```bash
# Check logs for errors
pm2 logs --lines 100

# Disable auto-restart temporarily
pm2 stop all
pm2 delete all

# Start manually to see errors
python -m uvicorn app.backend.main_v2:app --host localhost --port 8000
streamlit run app/frontend/app.py --server.port 8501
```

---

## Integration with Pipeline

The full pipeline (`run_full_pipeline.py`) automatically uses PM2 for the demo launch:

```bash
# Full pipeline with demo launch
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Only launch demo (requires trained model)
python scripts/run_full_pipeline.py --mode demo
```

If PM2 is not installed, the pipeline will:
1. Attempt to launch with PM2
2. Fail gracefully
3. Print manual instructions for running in separate terminals

---

## Alternative: Manual Launch (Without PM2)

If you prefer not to use PM2, you can run the demo manually:

### Terminal 1: Backend
```bash
python -m uvicorn app.backend.main_v2:app --host localhost --port 8000 --reload
```

### Terminal 2: Frontend
```bash
streamlit run app/frontend/app.py --server.port 8501
```

**Note**: This approach requires keeping both terminals open and doesn't provide auto-restart or centralized logging.

---

## Best Practices

### 1. Always Check Status
```bash
# Before starting
pm2 status

# After starting
pm2 status
pm2 logs --lines 20
```

### 2. Clean Shutdown
```bash
# Stop processes before closing terminal
pm2 stop all

# Or delete to remove from PM2 list
pm2 delete all
```

### 3. Monitor Resources
```bash
# Use interactive monitoring
pm2 monit

# Check memory usage
pm2 status
```

### 4. Regular Log Cleanup
```bash
# Clear old logs
pm2 flush

# Or manually delete log files
rm logs/*.log
```

### 5. Save Process List (Optional)
```bash
# Save current processes to resurrect after reboot
pm2 save

# Auto-start on system boot (optional)
pm2 startup
```

---

## Summary

### Quick Reference

| Task | Command |
|------|---------|
| **Install PM2** | `npm install -g pm2` |
| **Start Demo** | `python scripts/demo/run_demo_pm2.py` |
| **Stop Demo** | `pm2 stop all` |
| **View Status** | `pm2 status` |
| **View Logs** | `pm2 logs` |
| **Monitor** | `pm2 monit` |
| **Restart** | `pm2 restart all` |
| **Delete** | `pm2 delete all` |

### Recommended Workflow

1. **First Time Setup**:
   ```bash
   npm install -g pm2
   python scripts/run_full_pipeline.py --mode train-eval --training-mode quick
   ```

2. **Launch Demo**:
   ```bash
   python scripts/demo/run_demo_pm2.py
   ```

3. **Use the Demo**:
   - Open http://localhost:8501
   - Upload MRI images
   - View predictions and Grad-CAM

4. **Stop Demo**:
   ```bash
   pm2 stop all
   ```

---

## Additional Resources

- **PM2 Documentation**: https://pm2.keymetrics.io/docs/usage/quick-start/
- **PM2 GitHub**: https://github.com/Unitech/pm2
- **Ecosystem File Reference**: https://pm2.keymetrics.io/docs/usage/application-declaration/

---

**Last Updated**: December 8, 2025  
**Author**: SliceWise Team
