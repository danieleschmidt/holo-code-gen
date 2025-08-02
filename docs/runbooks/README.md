# Operational Runbooks

This directory contains operational runbooks for common scenarios and incident response procedures for Holo-Code-Gen.

## Available Runbooks

### Service Health
- [Service Down](./service-down.md) - When the main application is unavailable
- [High Error Rates](./high-error-rates.md) - Dealing with elevated error conditions
- [Performance Degradation](./performance-degradation.md) - Slow response times and optimization

### Resource Management
- [Memory Issues](./memory-issues.md) - High memory usage and leaks
- [Disk Space](./disk-space.md) - Storage management and cleanup
- [CPU Bottlenecks](./cpu-bottlenecks.md) - High CPU usage diagnosis

### Photonic-Specific Issues
- [Compilation Failures](./compilation-failures.md) - Photonic circuit compilation issues
- [Simulation Problems](./simulation-problems.md) - FDTD and photonic simulation troubleshooting
- [Library Loading Issues](./library-loading.md) - Photonic component library problems
- [GDS Export Failures](./gds-export-failures.md) - Layout generation and export issues

### Security & Compliance
- [Security Incidents](./security-incidents.md) - Security event response procedures
- [Unauthorized Access](./unauthorized-access.md) - Handling access violations
- [Data Breach Response](./data-breach-response.md) - Data security incident procedures

### Infrastructure
- [Container Issues](./container-issues.md) - Docker and container troubleshooting
- [Network Problems](./network-problems.md) - Connectivity and network issues
- [Database Problems](./database-problems.md) - Data persistence issues
- [Monitoring Failures](./monitoring-failures.md) - When monitoring systems fail

## Runbook Format

Each runbook follows a standard format:

### 1. Problem Description
Clear description of the issue, symptoms, and impact

### 2. Initial Assessment
Quick steps to assess severity and scope

### 3. Immediate Actions
Steps to mitigate immediate impact

### 4. Investigation
Detailed troubleshooting procedures

### 5. Resolution
Step-by-step resolution procedures

### 6. Prevention
Actions to prevent recurrence

### 7. Escalation
When and how to escalate

## Quick Reference

### Emergency Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| On-call Engineer | Slack: #holo-oncall | 24/7 |
| Lead Developer | email@company.com | Business hours |
| DevOps Team | Slack: #holo-ops | 24/7 |
| Security Team | security@company.com | 24/7 |

### Common Commands

```bash
# Check service status
docker ps | grep holo
docker logs holo-code-gen --tail 100

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Resource monitoring
docker stats holo-code-gen
htop

# Log analysis
tail -f /var/log/holo-code-gen/app.log
grep ERROR /var/log/holo-code-gen/app.log | tail -20
```

### Alert Severity Levels

| Level | Response Time | Escalation |
|-------|---------------|------------|
| Critical | 5 minutes | Immediate |
| High | 15 minutes | 30 minutes |
| Medium | 1 hour | 4 hours |
| Low | 4 hours | Next business day |

## Incident Response Process

### 1. Detection
- Automated alerts via Prometheus/Grafana
- User reports
- Monitoring dashboard anomalies
- Health check failures

### 2. Triage
- Assess impact and severity
- Identify affected systems/users
- Determine if this is a known issue
- Start incident response if needed

### 3. Response
- Follow appropriate runbook
- Communicate status to stakeholders
- Implement immediate mitigations
- Begin investigation and resolution

### 4. Resolution
- Apply permanent fix
- Verify resolution
- Monitor for recurrence
- Update incident status

### 5. Post-Incident
- Conduct post-mortem review
- Update runbooks and procedures
- Implement preventive measures
- Document lessons learned

## Tools and Resources

### Monitoring Tools
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application Logs**: `/var/log/holo-code-gen/`
- **Container Logs**: `docker logs holo-code-gen`

### Diagnostic Tools
- **Health Check**: `curl http://localhost:8000/health`
- **Metrics**: `curl http://localhost:8000/metrics`
- **System Resources**: `htop`, `iotop`, `df -h`
- **Network**: `netstat -tulpn`, `ss -tulpn`

### Recovery Tools
- **Service Restart**: `docker restart holo-code-gen`
- **Complete Reset**: `docker-compose down && docker-compose up -d`
- **Backup Restore**: See backup procedures
- **Rollback**: See deployment procedures

## Best Practices

### Documentation
- Keep runbooks current and tested
- Include exact commands and outputs
- Document decision points clearly
- Add troubleshooting flowcharts

### Testing
- Test runbooks regularly
- Simulate failure scenarios
- Update based on real incidents
- Train team on procedures

### Communication
- Use established communication channels
- Provide regular status updates
- Include relevant stakeholders
- Document all actions taken

### Follow-up
- Always conduct post-incident reviews
- Update procedures based on learnings
- Share knowledge with the team
- Implement preventive measures