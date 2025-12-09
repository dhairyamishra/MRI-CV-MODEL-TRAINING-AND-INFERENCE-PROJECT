module.exports = {
  apps: [
    {
      name: 'slicewise-backend',
      script: 'scripts/demo/start_backend.py',
      interpreter: 'pythonw',  // Use pythonw.exe (windowless) on Windows
      cwd: './',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      // Disable pidusage to avoid WMIC errors on Windows
      disable_logs: false,
      pmx: false,  // Disable PM2 metrics
      env: {
        PYTHONUNBUFFERED: '1'
      },
      error_file: './logs/backend-error.log',
      out_file: './logs/backend-out.log',
      log_file: './logs/backend-combined.log',
      time: true,
      merge_logs: true
    },
    {
      name: 'slicewise-frontend',
      script: 'scripts/demo/start_frontend.py',
      interpreter: 'pythonw',  // Use pythonw.exe (windowless) on Windows
      cwd: './',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      // Disable pidusage to avoid WMIC errors on Windows
      disable_logs: false,
      pmx: false,  // Disable PM2 metrics
      env: {
        PYTHONUNBUFFERED: '1'
      },
      error_file: './logs/frontend-error.log',
      out_file: './logs/frontend-out.log',
      log_file: './logs/frontend-combined.log',
      time: true,
      merge_logs: true
    }
  ]
};
