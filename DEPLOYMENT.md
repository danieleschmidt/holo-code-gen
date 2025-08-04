# Holo-Code-Gen Production Deployment Guide

## 🚀 Production-Ready Deployment

Holo-Code-Gen has successfully passed all quality gates and is ready for production deployment. This guide covers deployment configurations, monitoring setup, and operational procedures.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ or equivalent)
- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 10GB+ available space
- **Network**: HTTPS access for external integrations

### Required Dependencies
```bash
# Core scientific computing
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-pandas

# Optional: Advanced features
sudo apt install python3-networkx python3-h5py

# Development tools (optional)
sudo apt install python3-pytest python3-coverage
```

## 🏗️ Deployment Architectures

### 1. Standalone Deployment
```bash
# Clone and setup
git clone <repository-url>
cd holo-code-gen

# Install dependencies
pip3 install -r requirements.txt

# Initialize system
python3 -c "
from holo_code_gen.monitoring import initialize_monitoring
from holo_code_gen.security import initialize_security
from holo_code_gen.performance import initialize_performance

initialize_monitoring()
initialize_security() 
initialize_performance()
print('✅ Holo-Code-Gen initialized successfully')
"
```

### 2. Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-numpy python3-scipy python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python3 -c "from holo_code_gen import PhotonicCompiler; PhotonicCompiler()" || exit 1

EXPOSE 8000
CMD ["python3", "-m", "holo_code_gen.server"]
```

### 3. Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: holo-code-gen
  labels:
    app: holo-code-gen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: holo-code-gen
  template:
    metadata:
      labels:
        app: holo-code-gen
    spec:
      containers:
      - name: holo-code-gen
        image: holo-code-gen:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENABLE_METRICS
          value: "true"
        - name: MAX_WORKERS
          value: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: holo-code-gen-service
spec:
  selector:
    app: holo-code-gen
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## ⚙️ Configuration Management

### Environment Variables
```bash
# Core Configuration
export HOLO_LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
export HOLO_ENABLE_METRICS=true         # Enable metrics collection
export HOLO_ENABLE_CACHING=true         # Enable performance caching
export HOLO_MAX_WORKERS=4               # Number of parallel workers

# Security Configuration
export HOLO_MAX_CIRCUIT_COMPONENTS=10000 # Max components per circuit
export HOLO_MAX_FILE_SIZE_MB=100        # Max file upload size
export HOLO_ENABLE_SECURITY_AUDIT=true  # Enable security auditing

# Performance Configuration
export HOLO_CACHE_SIZE=1000             # Cache size (entries)
export HOLO_CACHE_TTL_SECONDS=3600      # Cache TTL in seconds
export HOLO_MAX_MEMORY_MB=4096          # Memory limit in MB

# Template Library
export HOLO_TEMPLATE_LIBRARY=imec_v2025_07  # Template library version
```

### Configuration File
Create `/etc/holo-code-gen/config.json`:
```json
{
  "compilation": {
    "template_library": "imec_v2025_07",
    "process": "SiN_220nm",
    "wavelength": 1550.0,
    "power_budget": 1000.0,
    "area_budget": 100.0
  },
  "security": {
    "enable_input_sanitization": true,
    "enable_path_validation": true,
    "max_circuit_components": 10000,
    "max_graph_nodes": 10000,
    "allowed_file_extensions": [".py", ".json", ".gds", ".spi"]
  },
  "performance": {
    "enable_caching": true,
    "cache_size": 1000,
    "cache_ttl_seconds": 3600,
    "enable_parallel_processing": true,
    "max_workers": 4,
    "enable_lazy_loading": true
  },
  "monitoring": {
    "enable_metrics": true,
    "export_interval": 60,
    "log_level": "INFO"
  }
}
```

## 📊 Monitoring & Observability

### Health Checks
```python
# Health check endpoint
from holo_code_gen.monitoring import get_health_checker

def health_check():
    health_checker = get_health_checker() 
    return health_checker.run_health_checks()

# Example response:
{
  "overall_healthy": true,
  "checks": {
    "template_library": {"status": "healthy"},
    "memory_usage": {"status": "healthy"},
    "cache_system": {"status": "healthy"}
  },
  "timestamp": "2025-08-04T04:44:44.000Z"
}
```

### Metrics Collection
```python
# Metrics available for monitoring systems
from holo_code_gen.performance import get_metrics_collector

metrics = get_metrics_collector()
exported_metrics = metrics.export_metrics()

# Key metrics to monitor:
# - compilation_duration_ms
# - cache_hit_rate
# - memory_usage_mb
# - error_rate
# - throughput_ops_per_second
```

### Prometheus Integration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'holo-code-gen'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard
Key panels to create:
- **Performance**: Compilation time, throughput, cache hit rate
- **Resources**: Memory usage, CPU utilization, disk I/O
- **Errors**: Error rate, failed compilations, security violations
- **Business**: Circuits compiled, users active, feature usage

## 🔐 Security Configuration

### SSL/TLS Setup
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name holo-code-gen.example.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Authentication
```python
# Example API key authentication
from holo_code_gen.security import get_security_auditor

def authenticate_request(api_key):
    auditor = get_security_auditor()
    auditor.audit_operation("authentication", "api_key", {"key_prefix": api_key[:8]})
    # Implement your authentication logic
    return validate_api_key(api_key)
```

## 🔄 Backup & Recovery

### Data Backup
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups/holo-code-gen/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r /etc/holo-code-gen "$BACKUP_DIR/config"

# Backup generated circuits (if persistent storage used)
cp -r /var/lib/holo-code-gen/circuits "$BACKUP_DIR/circuits"

# Backup logs
cp -r /var/log/holo-code-gen "$BACKUP_DIR/logs"

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" -C "$(dirname $BACKUP_DIR)" "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"

# Cleanup old backups (keep 7 days)
find /backups/holo-code-gen -name "*.tar.gz" -mtime +7 -delete
```

### Disaster Recovery
```bash
#!/bin/bash
# restore.sh
BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Stop service
systemctl stop holo-code-gen

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/

# Restore configuration
cp -r /tmp/*/config/* /etc/holo-code-gen/

# Restore data
cp -r /tmp/*/circuits/* /var/lib/holo-code-gen/circuits/

# Start service
systemctl start holo-code-gen

echo "Recovery completed from $BACKUP_FILE"
```

## 📈 Performance Tuning

### Memory Optimization
```python
# config.py
PERFORMANCE_CONFIG = {
    "cache_size": 2000,           # Increase for better caching
    "max_memory_mb": 8192,        # Set based on available RAM
    "enable_lazy_loading": True,  # Reduce memory footprint
    "gc_threshold": 0.8           # Trigger GC at 80% memory usage
}
```

### CPU Optimization
```python
# config.py
PARALLEL_CONFIG = {
    "max_workers": min(16, cpu_count() * 2),  # Optimal worker count
    "use_processes": False,                   # Use threads for I/O bound tasks
    "batch_size": 32,                        # Optimize batch processing
    "chunk_size": None                       # Auto-determine chunk size
}
```

### I/O Optimization
```python
# config.py
IO_CONFIG = {
    "enable_compression": True,    # Compress large files
    "buffer_size": 8192,          # Optimize file I/O buffer
    "async_export": True,         # Asynchronous file operations
    "temp_dir": "/tmp/holo"       # Fast temporary storage
}
```

## 🚨 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
python3 -c "
from holo_code_gen.performance import get_memory_manager
mgr = get_memory_manager()
print(mgr.check_memory_usage())
"

# Solutions:
# 1. Reduce cache size
# 2. Enable garbage collection
# 3. Increase memory limits
# 4. Scale horizontally
```

#### Slow Compilation
```bash
# Check performance metrics
python3 -c "
from holo_code_gen.performance import get_cache_manager
cache = get_cache_manager()
print(cache.get_stats())
"

# Solutions:
# 1. Increase cache size
# 2. Enable parallel processing
# 3. Optimize component templates
# 4. Use SSD storage
```

#### Security Violations
```bash
# Check security audit log
python3 -c "
from holo_code_gen.security import get_security_auditor
auditor = get_security_auditor()
violations = auditor.check_security_violations()
print(f'Found {len(violations)} violations')
for v in violations:
    print(f'- {v[\"type\"]}: {v[\"count\"]} occurrences')
"
```

### Log Analysis
```bash
# View application logs
tail -f /var/log/holo-code-gen/application.log

# Search for errors
grep "ERROR" /var/log/holo-code-gen/application.log | tail -20

# Monitor compilation performance
grep "compile.*Completed" /var/log/holo-code-gen/application.log | \
  awk '{print $NF}' | sed 's/[()ms]//g' | \
  sort -n | tail -10
```

## 🔄 Update & Maintenance

### Rolling Updates
```bash
#!/bin/bash
# rolling_update.sh
NEW_VERSION="$1"

# Health check function
health_check() {
    curl -f http://localhost:8000/health > /dev/null 2>&1
}

# Update process
echo "Starting rolling update to version $NEW_VERSION"

# Download new version
git fetch origin
git checkout "$NEW_VERSION"

# Install dependencies
pip3 install -r requirements.txt

# Restart with health checks
systemctl restart holo-code-gen

# Wait for service to be healthy
for i in {1..30}; do
    if health_check; then
        echo "✅ Update completed successfully"
        exit 0
    fi
    echo "Waiting for service to be healthy... ($i/30)"
    sleep 2
done

echo "❌ Update failed - service not healthy"
exit 1
```

### Maintenance Tasks
```bash
# Weekly maintenance script
#!/bin/bash

# Clear old cache entries
python3 -c "
from holo_code_gen.performance import get_cache_manager
cache = get_cache_manager()
cache.clear()
print('Cache cleared')
"

# Rotate logs
logrotate /etc/logrotate.d/holo-code-gen

# Update template libraries
python3 -c "
from holo_code_gen.templates import IMECLibrary
library = IMECLibrary()
print(f'Template library version: {library.version}')
"

# Run health checks
python3 -c "
from holo_code_gen.monitoring import get_health_checker
health = get_health_checker()
results = health.run_health_checks()
print(f'System health: {\"✅\" if results[\"overall_healthy\"] else \"❌\"}')
"
```

## ✅ Production Checklist

### Pre-Deployment
- [ ] All quality gates passed
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] SSL certificates installed
- [ ] Load balancer configured
- [ ] Health checks validated

### Post-Deployment
- [ ] Application responding to requests
- [ ] Metrics being collected
- [ ] Logs being generated
- [ ] Health checks passing
- [ ] Security monitoring active
- [ ] Backup jobs running
- [ ] Performance within SLA
- [ ] Error rates acceptable

### Ongoing Operations
- [ ] Daily health check review
- [ ] Weekly performance analysis
- [ ] Monthly security audit
- [ ] Quarterly dependency updates
- [ ] Annual disaster recovery test

## 📞 Support & Escalation

### Monitoring Alerts
- **Critical**: Service down, high error rate (>5%), memory exhaustion
- **Warning**: High latency (>1s), cache miss rate >50%, disk space low
- **Info**: New version deployed, configuration changed, maintenance started

### Escalation Matrix
1. **Level 1**: Automated recovery, restart services
2. **Level 2**: Engineering team notification
3. **Level 3**: Senior engineer involvement
4. **Level 4**: Emergency response team

---

**🎉 Congratulations! Holo-Code-Gen is now production-ready with enterprise-grade reliability, security, and performance.**