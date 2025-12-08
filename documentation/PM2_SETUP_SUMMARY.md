# PM2 Integration Summary

## What Changed?

Your demo application now uses **PM2** (Process Manager 2) for robust process management instead of Python's `subprocess` module. This solves the Windows compatibility issues you were experiencing.

## Problem Solved

**Before**: The demo would fail when launched from `run_full_pipeline.py` because subprocess management is unreliable on Windows. You had to run backend and frontend in separate terminals manually.

**After**: PM2 manages both processes reliably with auto-restart, centralized logging, and easy monitoring.

## New Files Created

1. **`configs/pm2-ecosystem/ecosystem.config.js`** - PM2 configuration for backend and frontend
2. **`scripts/demo/run_demo_pm2.py`** - New PM2-based launcher (378 lines)
3. **`documentation/PM2_DEMO_GUIDE.md`** - Comprehensive PM2 usage guide (488 lines)
4. **`PM2_SETUP_SUMMARY.md`** - This file

## Modified Files

1. **`scripts/run_full_pipeline.py`** - Updated `run_demo()` to use PM2 launcher
2. **`.gitignore`** - Added PM2 logs and process files
3. **`README.md`** - Added PM2 installation and usage instructions

## Installation

```bash
# Install PM2 globally (one-time setup)
npm install -g pm2

# Verify installation
pm2 --version
```

## Usage

### Quick Start

```bash
# Launch the demo (from project root)
python scripts/demo/run_demo_pm2.py
```

This will:
1. [OK] Check if PM2 is installed
2. [OK] Verify model checkpoint exists
3. [OK] Stop any existing demo processes
4. [OK] Start backend (FastAPI) on http://localhost:8000
5. [OK] Start frontend (Streamlit) on http://localhost:8501
6. [OK] Wait for health checks
7. [OK] Open browser automatically

### Managing the Demo

```bash
# View process status
pm2 status

# View logs (live tail)
pm2 logs

# Interactive monitoring
pm2 monit

# Stop the demo
pm2 stop all

# Stop and remove processes
pm2 delete all
```

### Full Pipeline Integration

The full pipeline now automatically uses PM2:

```bash
# Run full pipeline (data → training → evaluation → demo)
python scripts/run_full_pipeline.py --mode full --training-mode quick

# The demo will launch automatically with PM2 at the end
```

If PM2 is not installed, the pipeline will:
1. Attempt to launch with PM2
2. Fail gracefully
3. Print manual instructions for running in separate terminals

## Key Benefits

### ✅ Windows Compatibility
- No more subprocess issues on Windows
- Processes run reliably in the background

### ✅ Auto-Restart
- If backend or frontend crashes, PM2 automatically restarts it
- Configurable memory limits (2GB per process)

### ✅ Centralized Logging
- All logs stored in `logs/` directory
- Separate files for backend and frontend
- Easy to debug with `pm2 logs`

### ✅ Easy Monitoring
- `pm2 status` - Quick process overview
- `pm2 monit` - Interactive dashboard
- `pm2 logs` - Live log streaming

### ✅ Background Execution
- Processes run independently of terminal
- Can close terminal without stopping demo
- Survives SSH disconnections (if using remote server)

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

## Troubleshooting

### PM2 Not Found

If you get `pm2: command not found`:

```bash
# Install PM2
npm install -g pm2

# Check npm global bin path
npm config get prefix

# Add to PATH if needed (Windows)
# Add C:\Users\<YourUser>\AppData\Roaming\npm to PATH
```

### Port Already in Use

If ports 8000 or 8501 are already in use:

```bash
# Stop all PM2 processes
pm2 delete all

# Or kill processes manually (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Backend/Frontend Fails to Start

Check the logs:

```bash
# View backend logs
pm2 logs slicewise-backend --lines 50

# View frontend logs
pm2 logs slicewise-frontend --lines 50
```

Common issues:
- Model checkpoint not found → Run training first
- Missing dependencies → `pip install -r requirements.txt`
- Python environment → Activate correct environment before starting PM2

## Alternative: Manual Launch (Without PM2)

If you prefer not to use PM2, you can still run the demo manually:

**Terminal 1 (Backend):**
```bash
python -m uvicorn app.backend.main_v2:app --host localhost --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
streamlit run app/frontend/app.py --server.port 8501
```

**Note**: This requires keeping both terminals open and doesn't provide auto-restart or centralized logging.

## Documentation

For detailed information, see:
- **[PM2_DEMO_GUIDE.md](documentation/PM2_DEMO_GUIDE.md)** - Comprehensive PM2 usage guide
- **[README.md](README.md)** - Updated with PM2 installation and usage
- **PM2 Official Docs**: https://pm2.keymetrics.io/docs/usage/quick-start/

## Quick Reference

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

---

**Last Updated**: December 8, 2025  
**Author**: SliceWise Team
