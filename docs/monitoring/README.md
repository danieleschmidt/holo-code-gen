# Monitoring & Observability

This directory contains comprehensive monitoring and observability configuration for the Holo-Code-Gen project.

## Overview

Holo-Code-Gen includes a complete observability stack with:

- **Metrics Collection**: Prometheus for time-series metrics
- **Visualization**: Grafana dashboards for monitoring
- **Logging**: Structured logging with multiple output formats
- **Alerting**: Prometheus Alertmanager integration
- **Health Checks**: Application and dependency health monitoring
- **Tracing**: Distributed tracing for performance analysis

## Quick Start

### Local Development Monitoring

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

### Production Monitoring

```bash
# Deploy monitoring infrastructure
kubectl apply -f k8s/monitoring/

# Configure alerts
kubectl apply -f k8s/monitoring/alerts/
```

## Components

### Metrics Collection

#### Application Metrics

The application exposes metrics at `/metrics` endpoint:

| Metric | Type | Description |
|--------|------|-------------|
| `holo_code_gen_compilations_total` | Counter | Total number of compilations |
| `holo_code_gen_compilation_duration_seconds` | Histogram | Compilation time distribution |
| `holo_code_gen_active_simulations` | Gauge | Currently running simulations |
| `holo_code_gen_memory_usage_bytes` | Gauge | Memory usage |
| `holo_code_gen_errors_total` | Counter | Total application errors |

#### Custom Photonic Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `holo_photonic_components_loaded` | Gauge | Loaded photonic components |
| `holo_optimization_iterations_total` | Counter | Optimization iterations |
| `holo_gds_export_size_bytes` | Histogram | GDS file size distribution |
| `holo_simulation_accuracy_score` | Gauge | Simulation accuracy score |

### Dashboards

#### Main Overview Dashboard

- **System Health**: Service status, uptime, error rates
- **Performance**: Request rates, response times, throughput
- **Resources**: CPU, memory, disk usage
- **Photonic Metrics**: Compilation success rates, simulation performance

#### Photonic-Specific Dashboard

- **Component Library**: Template usage, load times
- **Optimization**: Algorithm performance, convergence rates
- **Simulation**: FDTD performance, accuracy metrics
- **Export**: GDS generation times, file sizes

### Alerting

#### Critical Alerts

- Service unavailability (> 1 minute)
- High error rates (> 10%)
- Resource exhaustion (> 90% memory/disk)
- Security events

#### Warning Alerts

- Performance degradation
- Simulation queue backlog
- Dependency failures
- Configuration drift

### Logging

#### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARNING**: Potentially harmful situations
- **ERROR**: Error events that don't stop execution
- **CRITICAL**: Serious errors that may abort execution

#### Log Categories

- **Application**: General application logs
- **Security**: Security-related events
- **Audit**: User actions and system changes
- **Performance**: Performance-related metrics
- **Simulation**: Photonic simulation logs

## Configuration

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'holo-code-gen'
    static_configs:
      - targets: ['holo-code-gen:8000']
```

### Grafana Configuration

```yaml
# monitoring/grafana/datasources/prometheus.yml
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
```

### Logging Configuration

```yaml
# monitoring/logging.yml
version: 1
formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
handlers:
  console:
    class: logging.StreamHandler
    formatter: json
```

## Health Checks

### Endpoints

| Endpoint | Purpose | Response Time |
|----------|---------|---------------|
| `/health` | Basic health check | < 1s |
| `/health/ready` | Readiness probe | < 5s |
| `/health/live` | Liveness probe | < 1s |
| `/health/deep` | Comprehensive check | < 30s |

### Health Check Components

- **Application Health**: Service responsiveness
- **Dependency Health**: External service availability
- **Resource Health**: Memory, disk, CPU availability
- **Feature Health**: Photonic library, optimization engine

## Operational Procedures

### Monitoring Playbook

#### Service Down
1. Check service logs: `docker logs holo-code-gen`
2. Verify resource availability
3. Check dependency health
4. Restart service if needed
5. Escalate if issue persists

#### High Error Rate
1. Identify error types from logs
2. Check recent deployments
3. Verify configuration changes
4. Roll back if necessary
5. Investigate root cause

#### Performance Degradation
1. Check resource utilization
2. Analyze slow requests
3. Review database performance
4. Check external dependencies
5. Scale resources if needed

### Troubleshooting

#### Common Issues

**No Metrics Available**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify application metrics endpoint
curl http://localhost:8000/metrics

# Check network connectivity
docker network inspect holo-network
```

**Dashboard Not Loading**
```bash
# Check Grafana logs
docker logs holo-grafana

# Verify datasource configuration
curl http://localhost:3000/api/datasources

# Test Prometheus connectivity
curl http://prometheus:9090/api/v1/query?query=up
```

**Alerts Not Firing**
```bash
# Check alerting rules
curl http://localhost:9090/api/v1/rules

# Verify Alertmanager connection
curl http://localhost:9090/api/v1/alertmanagers

# Test alert expression manually
curl "http://localhost:9090/api/v1/query?query=up{job='holo-code-gen'}"
```

## Advanced Configuration

### Custom Metrics

```python
# Example: Custom photonic metrics
from prometheus_client import Counter, Histogram, Gauge

COMPILATION_TIME = Histogram(
    'holo_compilation_duration_seconds',
    'Time spent compiling photonic circuits',
    ['template_type', 'optimization_level']
)

ACTIVE_SIMULATIONS = Gauge(
    'holo_active_simulations',
    'Number of currently running simulations',
    ['simulation_type']
)
```

### Distributed Tracing

```python
# Example: OpenTelemetry integration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=14268,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Log Aggregation

```yaml
# ELK Stack integration
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Best Practices

### Metric Naming

- Use descriptive names with units
- Include relevant labels for filtering
- Follow Prometheus naming conventions
- Avoid high-cardinality labels

### Alert Design

- Define clear severity levels
- Include actionable runbook links
- Set appropriate thresholds
- Test alert conditions regularly

### Dashboard Design

- Focus on business metrics
- Use appropriate visualizations
- Include context and annotations
- Design for different audiences

### Log Management

- Use structured logging (JSON)
- Include correlation IDs
- Log at appropriate levels
- Implement log rotation
- Consider log aggregation for scale