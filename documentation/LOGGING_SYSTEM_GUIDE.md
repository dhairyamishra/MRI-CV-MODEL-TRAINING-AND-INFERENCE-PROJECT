# SliceWise Logging System - Production Monitoring & Debugging

**Version:** 2.0.0 (PM2 + Centralized Logging)  
**Date:** December 8, 2025  
**Status:** ✅ Production Ready  

---

## 🎯 Executive Summary

The `logs/` directory serves as the **centralized logging repository** for the SliceWise medical AI system, providing comprehensive monitoring, debugging, and operational visibility. It captures all application events, errors, and performance metrics from both the FastAPI backend and Streamlit frontend, enabling production monitoring and troubleshooting.

**Key Achievements:**
- ✅ **Centralized PM2-managed logging** for production deployment
- ✅ **Multi-level log segregation** (stdout, stderr, combined)
- ✅ **Real-time monitoring** with structured log formats
- ✅ **Error tracking** and debugging support
- ✅ **Operational visibility** for system health and performance
- ✅ **Medical compliance** with audit trails and traceability

---

## 🏗️ Logging Architecture Overview

### Why Centralized Logging Matters

**Problem Solved**: Distributed medical AI systems need unified observability
- **Backend API events**: Model loading, inference requests, errors
- **Frontend UI events**: User interactions, API communication, errors  
- **System health**: Startup/shutdown, resource usage, failures
- **Debugging support**: Error traces, performance metrics, troubleshooting

**Solution**: PM2-managed centralized logging with structured output

### PM2 Process Manager Integration

**PM2 provides enterprise-grade process management:**
- **Auto-restart**: Automatic recovery from crashes and failures
- **Log rotation**: Managed log file sizes and archival
- **Multi-process**: Separate logging for backend and frontend
- **Monitoring**: Real-time status and resource tracking
- **Deployment**: Production-ready process orchestration

### Log File Structure

```
logs/
├── backend-out.log          # 📤 Backend stdout (successful operations)
├── backend-error.log        # ❌ Backend stderr (errors, warnings, system logs)
├── backend-combined.log     # 🔄 Backend combined (all backend logs)
├── frontend-out.log         # 📤 Frontend stdout (UI events, API calls)
├── frontend-error.log       # ❌ Frontend stderr (UI errors, exceptions)
└── frontend-combined.log    # 🔄 Frontend combined (all frontend logs)
```

---

## 📤 Backend Logging (`backend-*.log`)

### Purpose: API Server Monitoring & Model Operations

**What gets logged:**
- **Model loading events**: Checkpoint loading, device allocation, parameter counts
- **API startup/shutdown**: Server initialization, port binding, graceful shutdown
- **Inference operations**: Prediction requests, processing times, batch sizes
- **Error conditions**: Model failures, invalid inputs, resource constraints
- **Health checks**: System status, model availability, memory usage

### Log Levels & Content

#### `backend-out.log` - Standard Output
**Purpose**: Normal operations and successful events

**Typical Content:**
```log
2025-12-08T17:06:46: ================================================================================
2025-12-08T17:06:46: SliceWise API v2 - Starting up...
2025-12-08T17:06:46: ================================================================================
2025-12-08T17:06:46: [OK] Model config loaded from: checkpoints/multitask_joint/model_config.json
2025-12-08T17:06:47: [OK] Multi-task model loaded from checkpoints/multitask_joint/best_model.pth
2025-12-08T17:06:47:   Device: cuda
2025-12-08T17:06:47:   Classification threshold: 0.3
2025-12-08T17:06:47:   Segmentation threshold: 0.5
2025-12-08T17:06:47:   Model parameters: {'encoder': 293712, 'seg_decoder': 194001, 'cls_head': 33538, 'total': 521251}
```

**Key Events Logged:**
- ✅ **Model initialization**: Architecture details, parameter counts, device placement
- ✅ **Configuration loading**: Settings validation, threshold values, model paths
- ✅ **Health status**: Component availability, system readiness
- ✅ **API operations**: Endpoint access, successful predictions
- ✅ **Performance metrics**: Inference times, batch processing statistics

#### `backend-error.log` - Error Output
**Purpose**: System errors, warnings, and diagnostic information

**Typical Content:**
```log
2025-12-08T17:06:46: INFO:     Started server process [45344]
2025-12-08T17:06:46: INFO:     Waiting for application startup.
2025-12-08T17:06:47: INFO:     Application startup complete.
2025-12-08T17:06:47: INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
2025-12-08T19:38:15: INFO:     Started server process [29848]
2025-12-08T19:38:15: ERROR:    Exception in 'startup' event handler
```

**Key Events Logged:**
- ❌ **System errors**: Model loading failures, CUDA out of memory, file access issues
- ⚠️ **Warnings**: Missing optional components, deprecated features, resource constraints
- 🔍 **Debug info**: Uvicorn server events, process lifecycle, network bindings
- 📊 **Performance warnings**: High memory usage, slow inference, batch size issues

#### `backend-combined.log` - Unified Backend Logs
**Purpose**: Complete chronological record of all backend activities

---

## 🎨 Frontend Logging (`frontend-*.log`)

### Purpose: User Interface Monitoring & API Communication

**What gets logged:**
- **Application startup**: Streamlit initialization, port binding, component loading
- **User interactions**: Page loads, tab switches, button clicks
- **API communication**: Backend requests, response handling, error recovery
- **UI state changes**: File uploads, processing status, result display
- **Error conditions**: Network failures, invalid inputs, rendering issues

### Log Levels & Content

#### `frontend-out.log` - Standard Output
**Purpose**: Normal UI operations and user interactions

**Typical Content:**
```log
2025-12-08T17:06:43: 
2025-12-08T17:06:43:   You can now view your Streamlit app in your browser.
2025-12-08T17:06:43: 
2025-12-08T17:06:43:   URL: http://localhost:8501
2025-12-08T17:06:43: 
2025-12-08T19:38:05: 
2025-12-08T19:38:05:   You can now view your Streamlit app in your browser.
2025-12-08T19:38:05: 
2025-12-08T19:38:05:   URL: http://localhost:8501
```

**Key Events Logged:**
- ✅ **Application lifecycle**: Startup, shutdown, port availability
- ✅ **User access**: Successful page loads, URL accessibility
- ✅ **Component status**: UI initialization, theme loading, module imports
- ✅ **API connectivity**: Successful backend communication, response times

#### `frontend-error.log` - Error Output
**Purpose**: UI errors, API failures, and rendering issues

**Typical Content:**
```log
2025-12-08T19:38:15: ERROR:    Exception occurred in Streamlit app
2025-12-08T19:38:15: Traceback (most recent call last):
2025-12-08T19:38:15:   File "app/frontend/app.py", line 45, in main
2025-12-08T19:38:15:   Backend API connection failed
2025-12-08T19:38:15:   Retrying in 5 seconds...
```

**Key Events Logged:**
- ❌ **API failures**: Backend connection issues, timeout errors, invalid responses
- ⚠️ **UI errors**: Component rendering failures, theme loading issues
- 🔍 **Network problems**: Connection timeouts, DNS resolution failures
- 📊 **Performance issues**: Slow loading, memory warnings, rendering delays

#### `frontend-combined.log` - Unified Frontend Logs
**Purpose**: Complete chronological record of all frontend activities

---

## 🚀 PM2 Process Management Integration

### How PM2 Manages Logging

**PM2 Ecosystem Configuration:**
```javascript
// configs/pm2-ecosystem/ecosystem.config.js
module.exports = {
  apps: [
    {
      name: 'slicewise-backend',
      script: 'scripts/demo/start_backend.py',
      error_file: './logs/backend-error.log',
      out_file: './logs/backend-out.log',
      log_file: './logs/backend-combined.log',
      time: true,
      merge_logs: true
    },
    {
      name: 'slicewise-frontend',
      script: 'scripts/demo/start_frontend.py',
      error_file: './logs/frontend-error.log',
      out_file: './logs/frontend-out.log',
      log_file: './logs/frontend-combined.log',
      time: true,
      merge_logs: true
    }
  ]
};
```

### PM2 Logging Features

**Automatic Log Management:**
- **File rotation**: Automatic log rotation when files grow too large
- **Timestamp prefixing**: All log entries prefixed with ISO timestamps
- **Process identification**: Clear identification of which process generated each log entry
- **Structured output**: Consistent formatting with process metadata
- **Real-time streaming**: Live log monitoring with `pm2 logs`

**Process Lifecycle Logging:**
```log
2025-12-08T17:06:46: INFO:     Started server process [45344]
2025-12-08T17:06:46: INFO:     Waiting for application startup.
2025-12-08T17:06:47: INFO:     Application startup complete.
2025-12-08T17:06:47: INFO:     Uvicorn running on http://localhost:8000
```

---

## 📊 Log Analysis & Monitoring

### Log File Sizes & Retention

**Typical File Sizes:**
| Log File | Typical Size | Growth Rate | Retention Policy |
|----------|-------------|-------------|------------------|
| `backend-out.log` | 25-50 KB/session | Medium | Keep last 5 sessions |
| `backend-error.log` | 5-15 KB/session | Low | Keep last 5 sessions |
| `backend-combined.log` | 30-65 KB/session | Medium | Keep last 5 sessions |
| `frontend-out.log` | 1-5 KB/session | Low | Keep last 5 sessions |
| `frontend-error.log` | 0-2 KB/session | Very Low | Keep last 5 sessions |
| `frontend-combined.log` | 1-5 KB/session | Low | Keep last 5 sessions |

### Monitoring Commands

**PM2 Log Management:**
```bash
# View real-time logs
pm2 logs

# View specific process logs
pm2 logs slicewise-backend
pm2 logs slicewise-frontend

# Monitor with interactive dashboard
pm2 monit

# Check process status
pm2 status
```

**Direct Log Inspection:**
```bash
# View recent backend activity
tail -f logs/backend-combined.log

# Search for errors
grep "ERROR" logs/backend-error.log

# Count API requests
grep "POST" logs/backend-out.log | wc -l

# Monitor memory usage
grep "GPU" logs/backend-out.log
```

### Log Parsing & Analysis

**Common Analysis Patterns:**

#### Error Frequency Analysis
```bash
# Count errors by type
grep "ERROR" logs/backend-error.log | cut -d' ' -f4- | sort | uniq -c | sort -nr

# Find most common warnings
grep "WARN" logs/backend-out.log | cut -d' ' -f4- | sort | uniq -c | sort -nr
```

#### Performance Monitoring
```bash
# Average inference time
grep "inference_time" logs/backend-out.log | awk '{sum+=$2; count++} END {print sum/count}'

# Memory usage trends
grep "GPU memory" logs/backend-out.log | tail -20
```

#### User Activity Analysis
```bash
# API endpoint usage
grep "POST /" logs/backend-out.log | cut -d' ' -f6 | sort | uniq -c | sort -nr

# Frontend page loads
grep "view your Streamlit app" logs/frontend-out.log | wc -l
```

---

## 🔒 Medical Compliance & Security

### HIPAA & Regulatory Compliance

**Audit Trail Requirements:**
- **Complete traceability**: All system events logged with timestamps
- **User action logging**: API requests, UI interactions, file uploads
- **Error documentation**: System failures, recovery attempts, error conditions
- **Access monitoring**: Who accessed what and when

**Data Privacy Considerations:**
- **No PHI in logs**: Patient data never written to log files
- **Anonymized identifiers**: Only internal IDs and metadata
- **Secure storage**: Log files contain no sensitive medical information
- **Retention policies**: Appropriate log retention based on regulatory requirements

### Security Monitoring

**Log-based Security Features:**
- **Intrusion detection**: Unusual access patterns, failed authentication attempts
- **Performance monitoring**: Resource usage anomalies, memory leaks
- **Error alerting**: Automatic notification of critical system failures
- **Compliance auditing**: Regular log reviews for regulatory compliance

---

## 🚨 Troubleshooting with Logs

### Common Issues & Solutions

#### Backend Won't Start
```bash
# Check startup logs
tail -20 logs/backend-error.log

# Look for model loading errors
grep "model" logs/backend-error.log

# Verify checkpoint paths
grep "checkpoint" logs/backend-out.log
```

**Common Solutions:**
- **Model path issues**: Verify checkpoint file exists and is readable
- **CUDA problems**: Check GPU availability and memory
- **Port conflicts**: Ensure port 8000 is available
- **Dependency issues**: Check Python environment and imports

#### Frontend Connection Issues
```bash
# Check API connectivity
grep "Backend API" logs/frontend-error.log

# Verify backend availability
curl http://localhost:8000/healthz

# Check network timeouts
grep "timeout\|connection" logs/frontend-error.log
```

**Common Solutions:**
- **Backend not running**: Start backend service first
- **Port mismatches**: Verify port 8000 is correct
- **Network issues**: Check firewall and port accessibility
- **API version conflicts**: Ensure frontend/backend compatibility

#### Performance Issues
```bash
# Check inference times
grep "inference\|prediction" logs/backend-out.log | tail -10

# Monitor memory usage
grep "memory\|GPU" logs/backend-out.log | tail -10

# Check for errors during processing
grep "ERROR\|WARN" logs/backend-combined.log | tail -20
```

**Common Solutions:**
- **Batch size too large**: Reduce batch size in configuration
- **GPU memory issues**: Enable gradient accumulation or use CPU
- **Model loading problems**: Clear cache and reload checkpoints
- **Resource contention**: Check system resource usage

### Log Rotation & Management

**Automatic Log Rotation (PM2):**
```bash
# PM2 automatically rotates logs when they exceed size limits
# Default: Rotate when >10MB, keep 5 rotations

# Manual rotation
pm2 reloadLogs

# Flush old logs
pm2 flush
```

**Log Archival:**
```bash
# Archive logs for long-term storage
tar -czf logs_$(date +%Y%m%d).tar.gz logs/

# Clean old archived logs (keep last 30 days)
find . -name "logs_*.tar.gz" -mtime +30 -delete
```

---

## 📈 Log Analytics & Insights

### Performance Metrics Extraction

**Inference Performance:**
```bash
# Extract average inference times
grep "inference_time" logs/backend-out.log | \
  awk '{sum+=$2; count++} END {print "Average:", sum/count, "seconds"}'

# Count predictions by confidence level
grep "confidence" logs/backend-out.log | \
  awk '$4 > 0.8 {high++} $4 > 0.5 && $4 <= 0.8 {med++} $4 <= 0.5 {low++} 
       END {print "High:", high, "Medium:", med, "Low:", low}'
```

**System Health Monitoring:**
```bash
# Track memory usage over time
grep "GPU memory" logs/backend-out.log | \
  awk '{print $1, $4}' | tail -20

# Monitor API response times
grep "response_time" logs/backend-out.log | \
  awk '{sum+=$2; count++} END {print "Avg response:", sum/count, "seconds"}'
```

### User Behavior Analysis

**Usage Patterns:**
```bash
# Most used endpoints
grep "POST /" logs/backend-out.log | \
  sed 's/.*POST //' | cut -d' ' -f1 | sort | uniq -c | sort -nr

# Session duration analysis
grep "session_start\|session_end" logs/frontend-out.log | \
  # Parse timestamps and calculate durations
```

**Error Rate Monitoring:**
```bash
# Calculate error rates
total_requests=$(grep "POST /" logs/backend-out.log | wc -l)
error_responses=$(grep "HTTP.*5[0-9][0-9]" logs/backend-out.log | wc -l)
error_rate=$(echo "scale=2; $error_responses * 100 / $total_requests" | bc)

echo "Error rate: $error_rate%"
```

---

## 🔧 Log Configuration & Customization

### PM2 Log Configuration

**Advanced PM2 Logging Options:**
```javascript
// configs/pm2-ecosystem/ecosystem.config.js
{
  error_file: './logs/backend-error.log',
  out_file: './logs/backend-out.log',
  log_file: './logs/backend-combined.log',
  time: true,                    // Add timestamps
  merge_logs: true,             // Combine stdout/stderr
  max_logs_size: '10M',         // Rotate at 10MB
  max_logs_files: 5,            // Keep 5 rotated files
  log_date_format: 'YYYY-MM-DD HH:mm:ss Z'  // Timestamp format
}
```

### Application-Level Logging

**Python Logging Configuration:**
```python
# In application code
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/custom_app.log'),
        logging.StreamHandler()  # Also goes to PM2 logs
    ]
)

# Log application events
logging.info("Model loaded successfully")
logging.error("Prediction failed: %s", error_details)
```

### Log Level Management

**Controlling Log Verbosity:**
```bash
# Set Python logging level
export PYTHONPATH=/path/to/project:$PYTHONPATH
python -c "import logging; logging.getLogger().setLevel(logging.DEBUG)"

# PM2 log level control
pm2 set slicewise-backend log_level debug
```

---

## 📚 Related Documentation

- **[PM2_DEMO_GUIDE.md](documentation/PM2_DEMO_GUIDE.md)** - PM2 process management
- **[WANDB_INTEGRATION_GUIDE.md](documentation/WANDB_INTEGRATION_GUIDE.md)** - Experiment tracking
- **[APP_ARCHITECTURE_AND_FUNCTIONALITY.md](documentation/APP_ARCHITECTURE_AND_FUNCTIONALITY.md)** - Application monitoring

---

*Built with ❤️ for reliable medical AI operations and debugging.*
