# Release Process Documentation

## Overview

Holo-Code-Gen follows semantic versioning with automated release management using semantic-release. This document outlines the complete release workflow from development to production deployment.

## Semantic Versioning

### Version Format

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

### Version Increment Rules

| Change Type | Version Impact | Example |
|-------------|----------------|---------|
| Breaking changes | MAJOR | 1.0.0 → 2.0.0 |
| New features | MINOR | 1.0.0 → 1.1.0 |
| Bug fixes | PATCH | 1.0.0 → 1.0.1 |
| Pre-release | PRERELEASE | 1.0.0 → 1.1.0-beta.1 |

### Commit Message Format

Following Angular commit convention:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types

- `feat`: New feature (MINOR bump)
- `fix`: Bug fix (PATCH bump)
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes

#### Breaking Changes

```
feat: redesign photonic compiler API

BREAKING CHANGE: PhotonicCompiler.compile() now returns PhotonicCircuit instead of dict
```

#### Examples

```bash
# Feature addition (minor bump)
git commit -m "feat(compiler): add support for multi-wavelength optimization"

# Bug fix (patch bump)
git commit -m "fix(simulation): resolve memory leak in FDTD solver"

# Breaking change (major bump)
git commit -m "feat!: redesign template library interface

BREAKING CHANGE: TemplateLibrary.get_component() now requires explicit version parameter"
```

## Release Workflow

### Automated Release Process

1. **Commit Analysis**: semantic-release analyzes commit messages
2. **Version Calculation**: Determines next version based on changes
3. **Changelog Generation**: Auto-generates CHANGELOG.md
4. **Version Update**: Updates version in `pyproject.toml` and `__init__.py`
5. **Build Artifacts**: Creates distribution packages
6. **GitHub Release**: Creates GitHub release with assets
7. **PyPI Publishing**: Uploads packages to PyPI (when configured)

### Manual Release Trigger

```bash
# Install semantic-release
npm install -g semantic-release @semantic-release/changelog @semantic-release/git

# Dry run (preview changes)
semantic-release --dry-run

# Execute release
semantic-release
```

### Branch Strategy

#### Main Branch (`main`)
- Production-ready code
- All releases are tagged from this branch
- Protected with required PR reviews
- Automated releases on push

#### Development Branch (`develop`)
- Integration branch for features
- Beta releases can be created
- Merges to main trigger releases

#### Feature Branches (`feature/*`)
- Individual feature development
- No automated releases
- Must pass all CI checks before merge

#### Hotfix Branches (`hotfix/*`)
- Critical production fixes
- Can trigger patch releases directly
- Merged to both main and develop

### Release Configuration

The `.releaserc.json` configuration:

```json
{
  "branches": [
    "main",
    {
      "name": "beta",
      "prerelease": true
    }
  ],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    "@semantic-release/exec",
    "@semantic-release/github",
    "@semantic-release/git"
  ]
}
```

## Pre-release Process

### Quality Gates

Before any release, the following checks must pass:

1. **Unit Tests**: Full test suite execution
   ```bash
   make test
   ```

2. **Integration Tests**: End-to-end functionality
   ```bash
   make test-photonic
   ```

3. **Code Quality**: Linting and type checking
   ```bash
   make lint
   make type-check
   ```

4. **Security Scan**: Vulnerability assessment
   ```bash
   safety check
   bandit -r holo_code_gen/
   ```

5. **Documentation**: Ensure docs are current
   ```bash
   make docs
   ```

### Pre-release Checklist

- [ ] All CI checks passing
- [ ] Documentation updated
- [ ] CHANGELOG preview reviewed
- [ ] Breaking changes documented
- [ ] Migration guide prepared (if needed)
- [ ] Security vulnerabilities addressed
- [ ] Performance benchmarks stable
- [ ] Integration tests with foundry PDKs
- [ ] Example code updated

## Release Types

### Major Release (X.0.0)

Major releases include breaking changes and require special attention:

#### Preparation
1. **Migration Guide**: Document breaking changes
2. **Deprecation Warnings**: Added in previous minor versions
3. **Compatibility Matrix**: Update supported versions
4. **Beta Testing**: Extended beta period with early adopters

#### Communication
- Blog post announcement
- Email to mailing list
- GitHub Discussions post
- Update documentation website

#### Example: v2.0.0
```
# Breaking changes
- PhotonicCompiler API redesign
- Template library format update
- Minimum Python 3.10 requirement

# New features
- GPU-accelerated simulation
- Advanced optimization algorithms
- Real-time collaboration features
```

### Minor Release (x.Y.0)

Minor releases add new features while maintaining compatibility:

#### Features
- New photonic components
- Enhanced optimization algorithms
- Additional export formats
- Performance improvements

#### Example: v1.5.0
```
# New features
- Support for SiN platform templates
- Multi-objective optimization
- Batch processing capabilities

# Improvements
- 30% faster compilation
- Enhanced error messages
- Better memory efficiency
```

### Patch Release (x.y.Z)

Patch releases contain bug fixes and security updates:

#### Criteria
- Critical bug fixes
- Security vulnerabilities
- Documentation corrections
- Dependency updates

#### Example: v1.4.3
```
# Bug fixes
- Fix memory leak in ring resonator simulation
- Correct phase shifter power calculation
- Resolve GDS export encoding issue

# Security
- Update vulnerable dependencies
- Patch authentication bypass
```

## Beta Releases

### Beta Process

1. **Beta Branch**: Create from develop
   ```bash
   git checkout -b beta develop
   git push origin beta
   ```

2. **Auto-versioning**: Beta versions follow pattern `x.y.z-beta.n`

3. **Beta Testing**: 
   - Internal testing with core team
   - External testing with select users
   - Automated testing in staging environment

4. **Feedback Integration**: Address issues before main release

### Beta Distribution

```bash
# Install beta version
pip install holo-code-gen==1.5.0-beta.1

# Docker beta image
docker pull holo-code-gen:1.5.0-beta.1
```

## Hotfix Process

### Critical Issue Response

1. **Assessment**: Determine severity and impact
2. **Hotfix Branch**: Create from main
   ```bash
   git checkout -b hotfix/critical-security-fix main
   ```

3. **Fix Development**: Minimal changes to address issue
4. **Testing**: Focused testing on fix area
5. **Release**: Immediate patch release
6. **Backport**: Merge back to develop

### Hotfix Example

```bash
# Create hotfix branch
git checkout -b hotfix/memory-leak-fix main

# Make minimal fix
# ... implement fix ...

# Commit with fix type
git commit -m "fix: resolve memory leak in simulation engine

Critical memory leak affecting long-running simulations.
Impact: High - affects production workloads
Solution: Proper cleanup of simulation contexts"

# Push and create PR
git push origin hotfix/memory-leak-fix
# Create PR to main for immediate release
```

## Distribution

### Package Distribution

#### PyPI
```bash
# Build packages
python -m build

# Check packages
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

#### Conda
```bash
# Create conda package
conda-build recipe/

# Upload to conda-forge
# (Automated via conda-forge feedstock)
```

#### Docker Registry
```bash
# Build and tag
docker build -t holo-code-gen:1.5.0 .
docker tag holo-code-gen:1.5.0 holo-code-gen:latest

# Push to registry
docker push holo-code-gen:1.5.0
docker push holo-code-gen:latest
```

### Release Artifacts

Each release includes:

1. **Source Distribution** (`*.tar.gz`)
2. **Python Wheel** (`*.whl`)
3. **Docker Images** (multiple architectures)
4. **Documentation** (HTML and PDF)
5. **Example Projects** (ZIP archive)
6. **Checksums** (SHA256 hashes)

## Post-release Process

### Verification

1. **Installation Test**: Verify packages install correctly
   ```bash
   pip install holo-code-gen==1.5.0
   python -c "import holo_code_gen; print(holo_code_gen.__version__)"
   ```

2. **Smoke Tests**: Basic functionality verification
   ```bash
   holo-code-gen --version
   python -m holo_code_gen.examples.basic_nn
   ```

3. **Documentation**: Ensure docs website updated
4. **Docker Images**: Verify multi-arch images work

### Communication

1. **Release Notes**: Publish detailed release notes
2. **Social Media**: Announce on Twitter/LinkedIn
3. **Community**: Update Discord/Slack channels
4. **Metrics**: Monitor download/usage metrics

### Monitoring

Post-release monitoring includes:

- **Error Tracking**: Monitor error rates
- **Performance**: Check performance regressions
- **Usage Analytics**: Track feature adoption
- **User Feedback**: Monitor issues and discussions

## Rollback Procedures

### Emergency Rollback

If critical issues are discovered:

1. **Immediate**: Remove problematic release from PyPI
   ```bash
   # Contact PyPI support for emergency removal
   ```

2. **Docker**: Remove affected tags
   ```bash
   # Remove from registry (if possible)
   # Update latest tag to previous version
   ```

3. **Communication**: Immediate notification to users
4. **Hotfix**: Prepare emergency fix release

### Partial Rollback

For less critical issues:

1. **Advisory**: Publish security advisory
2. **Fixed Version**: Release patched version quickly
3. **Documentation**: Update with workarounds
4. **Support**: Provide migration assistance

## Automation Setup

### GitHub Actions Integration

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    branches: [main]
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-node@v4
      - run: npm install -g semantic-release
      - run: semantic-release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
```

### Release Notifications

```yaml
# Slack notification on release
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: "Holo-Code-Gen ${{ steps.semantic.outputs.new_release_version }} released!"
```