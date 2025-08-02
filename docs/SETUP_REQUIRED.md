# Manual Setup Required

Due to GitHub App permission limitations, the following components require manual setup by repository maintainers after the SDLC implementation is complete.

## 🔧 GitHub Workflows Setup

### 1. Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Templates

Copy the following workflow files from `docs/workflows/examples/` to `.github/workflows/`:

| Template File | Target File | Description |
|---------------|-------------|-------------|
| `docs/workflows/examples/ci.yml` | `.github/workflows/ci.yml` | Continuous Integration |
| `docs/workflows/examples/cd.yml` | `.github/workflows/cd.yml` | Continuous Deployment |
| `docs/workflows/examples/security-scan.yml` | `.github/workflows/security-scan.yml` | Security Scanning |
| `docs/workflows/examples/dependency-update.yml` | `.github/workflows/dependency-update.yml` | Dependency Updates |

### 3. Configure Repository Secrets

Add the following secrets in GitHub repository settings:

#### Required Secrets
- `PYPI_API_TOKEN`: For PyPI package publishing
- `CODECOV_TOKEN`: For code coverage reporting

#### Optional Secrets (if applicable)
- `PDK_ACCESS_TOKEN`: For foundry PDK access
- `SLACK_WEBHOOK_URL`: For Slack notifications
- `DISCORD_WEBHOOK_URL`: For Discord notifications

### 4. Branch Protection Rules

Configure branch protection for `main` branch:

```yaml
Protection Rules:
  - Require pull request reviews: ✅ (1 reviewer minimum)
  - Require status checks to pass: ✅
    - Required checks:
      - CI / Test Suite
      - CI / Code Quality & Security
      - CI / Photonic Component Tests
      - CI / Documentation
  - Require branches to be up to date: ✅
  - Require linear history: ✅
  - Include administrators: ✅
  - Allow force pushes: ❌
  - Allow deletions: ❌
```

## 🔒 Security Configuration

### 1. Enable Security Features

In repository settings → Security, enable:

- [x] Dependency graph
- [x] Dependabot alerts
- [x] Dependabot security updates
- [x] Secret scanning
- [x] Code scanning (CodeQL)

### 2. Configure Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 10
    reviewers:
      - "your-team"
    labels:
      - "dependencies"
      - "automated"
```

### 3. Security Policy

Ensure `SECURITY.md` is properly configured and linked in repository settings.

## 📊 Monitoring Integration

### 1. Codecov Integration

1. Go to [Codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Copy the upload token
4. Add as `CODECOV_TOKEN` secret

### 2. External Monitoring

If using external monitoring services:

- **Grafana Cloud**: Configure webhook for deployment notifications
- **DataDog**: Set up GitHub integration
- **Sentry**: Add error tracking integration

## 🚀 Deployment Configuration

### 1. Container Registry

For GitHub Container Registry (ghcr.io):

1. Enable "Improved container support" in repository settings
2. Ensure workflow has `packages: write` permission
3. Configure package visibility settings

### 2. PyPI Publishing

1. Create PyPI account and verify email
2. Generate API token with appropriate scope
3. Add token as `PYPI_API_TOKEN` secret

### 3. Environment Setup

Create deployment environments in repository settings:

#### Staging Environment
- **Name**: `staging`
- **Protection rules**: No restrictions
- **Environment secrets**: Staging-specific credentials

#### Production Environment
- **Name**: `production`
- **Protection rules**: Required reviewers
- **Environment secrets**: Production credentials

#### Production Approval Environment
- **Name**: `production-approval`
- **Protection rules**: Required reviewers (different from production)

## 🏷️ Issue Templates

Create `.github/ISSUE_TEMPLATE/` directory with templates:

### Bug Report Template
```yaml
name: Bug Report
about: Create a report to help us improve
title: "[BUG] "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: version
    attributes:
      label: Version
      description: What version of holo-code-gen are you running?
    validations:
      required: true
  # Add more fields as needed
```

### Feature Request Template
```yaml
name: Feature Request
about: Suggest an idea for this project
title: "[FEATURE] "
labels: ["enhancement", "triage"]
# Add fields as needed
```

## 📋 Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Photonic simulation tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated (if applicable)

## Photonic-Specific Checklist (if applicable)
- [ ] PDK compatibility verified
- [ ] GDS generation tested
- [ ] Simulation accuracy validated
- [ ] Performance benchmarks run
```

## 🔗 Integrations

### 1. Read the Docs

1. Go to [Read the Docs](https://readthedocs.org)
2. Import your repository
3. Configure build settings
4. Set up webhook integration

### 2. Code Quality Services

#### SonarCloud
1. Connect repository to SonarCloud
2. Configure quality gates
3. Add badges to README

#### Codacy
1. Add repository to Codacy
2. Configure code patterns
3. Set up PR integration

## 📈 Analytics and Monitoring

### 1. GitHub Insights

Enable and configure:
- Pulse
- Contributors
- Community
- Traffic
- Dependency graph

### 2. Package Analytics

For PyPI packages:
- Monitor download statistics
- Track version adoption
- Analyze geographic distribution

## ⚙️ Repository Settings

### General Settings
- **Default branch**: `main`
- **Template repository**: Disabled
- **Issues**: Enabled
- **Wiki**: Disabled (use docs/ instead)
- **Discussions**: Enabled
- **Sponsorships**: As appropriate

### Merge Button Settings
- **Allow merge commits**: ✅
- **Allow squash merging**: ✅
- **Allow rebase merging**: ✅
- **Automatically delete head branches**: ✅

### Code Security Settings
- **Private vulnerability reporting**: Enabled
- **Dependency graph**: Enabled
- **Dependabot alerts**: Enabled
- **Dependabot security updates**: Enabled
- **Secret scanning**: Enabled

## 🚨 Validation Checklist

After completing setup, verify:

- [ ] All workflows run successfully
- [ ] Branch protection rules are active
- [ ] Security features are enabled
- [ ] Deployment environments configured
- [ ] Monitoring integrations working
- [ ] Issue/PR templates functional
- [ ] External integrations connected
- [ ] Team permissions configured
- [ ] Repository settings optimized

## 📞 Support

If you encounter issues during setup:

1. Check GitHub documentation for the specific feature
2. Review workflow logs for error details
3. Consult the troubleshooting section in relevant documentation
4. Open an issue with the `setup-help` label

## 🔄 Maintenance

Regular maintenance tasks:

- **Weekly**: Review security alerts and dependency updates
- **Monthly**: Audit repository permissions and access
- **Quarterly**: Review and update workflow configurations
- **Annually**: Complete security audit and update procedures