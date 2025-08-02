# Runbook: Service Down

## Problem Description

The Holo-Code-Gen service is unavailable or unresponsive, affecting all users and photonic circuit compilation/simulation capabilities.

**Symptoms:**
- Health check endpoints return 5xx errors or timeout
- Users cannot access the application
- Compilation and simulation requests fail
- Monitoring shows service as down

**Impact:** High - Complete service unavailability

## Initial Assessment (< 2 minutes)

### 1. Verify the Issue
```bash
# Check if service responds
curl -f http://localhost:8000/health
# Expected: 200 OK with {"status": "healthy"}

# Check if container is running
docker ps | grep holo-code-gen
# Expected: Container in "Up" status

# Quick log check
docker logs holo-code-gen --tail 20
# Look for recent error messages
```

### 2. Assess Scope
- [ ] Single instance down vs. all instances
- [ ] Partial functionality vs. complete outage
- [ ] Recent deployments or changes
- [ ] External dependency failures

### 3. Check Infrastructure
```bash
# System resources
df -h  # Check disk space
free -h  # Check memory
top  # Check CPU and processes

# Network connectivity
ping prometheus
ping grafana
```

## Immediate Actions (< 5 minutes)

### 1. Attempt Quick Recovery

#### Container Restart
```bash
# Restart the main service
docker restart holo-code-gen

# Wait 30 seconds and test
sleep 30
curl -f http://localhost:8000/health

# Check logs for startup errors
docker logs holo-code-gen --tail 50
```

#### Service Stack Restart
```bash
# If container restart fails, restart the entire stack
docker-compose down
sleep 10
docker-compose up -d

# Monitor startup
docker-compose logs -f holo-code-gen
```

### 2. Verify Recovery
```bash
# Test health endpoint
curl -f http://localhost:8000/health

# Test basic functionality
curl -f http://localhost:8000/api/v1/status

# Check metrics endpoint
curl -f http://localhost:8000/metrics | head -20
```

## Detailed Investigation

### 1. Log Analysis

#### Application Logs
```bash
# Check for errors in application logs
docker logs holo-code-gen | grep -E "(ERROR|CRITICAL|FATAL)" | tail -20

# Look for startup issues
docker logs holo-code-gen | grep -E "(startup|initialization)" | tail -10

# Check for specific error patterns
docker logs holo-code-gen | grep -E "(OutOfMemory|ConnectionError|TimeoutError)" | tail -10
```

#### System Logs
```bash
# Check system logs for container issues
journalctl -u docker.service | tail -20

# Check for resource exhaustion
dmesg | grep -E "(killed|OOM|out of memory)" | tail -10

# Check disk I/O issues
dmesg | grep -E "(I/O error|read error|write failed)" | tail -10
```

### 2. Resource Investigation

#### Memory Issues
```bash
# Check memory usage
free -h
docker stats holo-code-gen --no-stream

# Check for memory leaks in recent history
# (Requires monitoring data)
curl -s "http://prometheus:9090/api/v1/query?query=process_resident_memory_bytes{job='holo-code-gen'}"
```

#### Disk Space Issues
```bash
# Check disk usage
df -h

# Check specific directories
du -sh /var/lib/docker
du -sh /app/data
du -sh /app/logs

# Check for large log files
find /app/logs -name "*.log" -size +100M -ls
```

#### CPU Issues
```bash
# Check CPU usage
top -p $(pgrep -f holo-code-gen)

# Check for high CPU processes
ps aux | sort -nr -k 3 | head -10
```

### 3. Configuration Issues

#### Environment Variables
```bash
# Check container environment
docker exec holo-code-gen env | grep HOLO_

# Verify configuration files
docker exec holo-code-gen cat /app/config/settings.yaml
```

#### Dependencies
```bash
# Check external dependencies
curl -f http://prometheus:9090/-/healthy
curl -f http://grafana:3000/api/health

# Check database connectivity (if applicable)
# docker exec holo-code-gen python -c "from app.database import test_connection; test_connection()"
```

### 4. Network Issues

#### Container Networking
```bash
# Check container network
docker network ls
docker network inspect holo-network

# Test inter-container connectivity
docker exec holo-code-gen ping prometheus
docker exec holo-code-gen ping grafana
```

#### Port Binding
```bash
# Check port bindings
docker port holo-code-gen
netstat -tulpn | grep :8000
```

## Resolution Procedures

### 1. Configuration Fix
If configuration issues are identified:

```bash
# Update configuration
vim docker-compose.yml
# or
vim /app/config/settings.yaml

# Apply changes
docker-compose down
docker-compose up -d

# Verify fix
curl -f http://localhost:8000/health
```

### 2. Resource Recovery
If resource exhaustion is the cause:

```bash
# Clean up disk space
docker system prune -f
docker volume prune -f

# Increase resource limits
# Edit docker-compose.yml to increase memory/CPU limits
docker-compose down
docker-compose up -d
```

### 3. Application Recovery
If application-specific issues:

```bash
# Clear application cache
docker exec holo-code-gen rm -rf /app/cache/*

# Reset application state
docker exec holo-code-gen python -c "from app.utils import reset_state; reset_state()"

# Restart with fresh state
docker restart holo-code-gen
```

### 4. Full Recovery
If partial fixes don't work:

```bash
# Complete environment reset
docker-compose down -v  # WARNING: Removes volumes
docker system prune -a -f
docker-compose pull
docker-compose up -d

# Restore data from backup (if needed)
# See backup restoration procedures
```

## Verification Steps

### 1. Health Checks
```bash
# Basic health
curl -f http://localhost:8000/health
# Expected: {"status": "healthy", "version": "0.1.0", "uptime": "..."}

# Readiness check
curl -f http://localhost:8000/health/ready
# Expected: {"status": "ready", "dependencies": {...}}

# Deep health check
curl -f http://localhost:8000/health/deep
# Expected: Detailed component status
```

### 2. Functionality Tests
```bash
# Test compilation endpoint
curl -X POST http://localhost:8000/api/v1/compile \
  -H "Content-Type: application/json" \
  -d '{"test": true}'

# Test metrics collection
curl -f http://localhost:8000/metrics | grep holo_code_gen

# Test simulation endpoint (if available)
curl -X GET http://localhost:8000/api/v1/simulations/status
```

### 3. Performance Validation
```bash
# Check response times
time curl -f http://localhost:8000/health

# Monitor resource usage
docker stats holo-code-gen --no-stream

# Check for error rates
# (Requires monitoring setup)
```

## Prevention Measures

### 1. Monitoring Improvements
- Set up proactive alerting for resource usage
- Implement log aggregation and analysis
- Add custom health checks for critical components
- Monitor external dependencies

### 2. Resource Management
- Implement proper resource limits
- Set up log rotation
- Configure automatic cleanup jobs
- Monitor disk usage trends

### 3. Deployment Practices
- Implement proper health checks in deployment
- Use blue-green deployments
- Add deployment rollback procedures
- Test in staging environment first

### 4. Documentation Updates
- Update this runbook with new findings
- Document configuration changes
- Create troubleshooting flowcharts
- Train team on procedures

## Escalation

### When to Escalate
- Service remains down after 30 minutes
- Resource issues cannot be resolved
- Data corruption is suspected
- Security breach is suspected
- Multiple services are affected

### Escalation Contacts
1. **Lead Developer**: developer@company.com
2. **DevOps Team**: Slack #holo-ops
3. **Infrastructure Team**: infrastructure@company.com
4. **Management**: If business impact > 1 hour

### Escalation Information to Provide
- Timeline of the incident
- Actions taken so far
- Current status and symptoms
- Impact assessment
- Relevant log excerpts
- Resource utilization data

## Post-Incident Actions

### 1. Immediate
- [ ] Verify complete service recovery
- [ ] Monitor for 1 hour for stability
- [ ] Update stakeholders on resolution
- [ ] Document timeline and actions taken

### 2. Short-term (24 hours)
- [ ] Analyze root cause
- [ ] Update monitoring/alerting if needed
- [ ] Review and update procedures
- [ ] Plan preventive measures

### 3. Long-term (1 week)
- [ ] Conduct post-mortem meeting
- [ ] Implement systemic improvements
- [ ] Update documentation
- [ ] Review SLA/SLO compliance

## Related Runbooks
- [High Error Rates](./high-error-rates.md)
- [Performance Degradation](./performance-degradation.md)
- [Memory Issues](./memory-issues.md)
- [Container Issues](./container-issues.md)