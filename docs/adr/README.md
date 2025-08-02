# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Holo-Code-Gen project. ADRs document important architectural decisions made during development, including the context, decision, and consequences.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-template-based-photonic-design.md) | Template-Based Photonic Component Design | ACCEPTED | 2025-08-02 |

## ADR Template

When creating new ADRs, use the following template:

```markdown
# ADR-XXXX: [Title]

## Status
**[PROPOSED/ACCEPTED/DEPRECATED/SUPERSEDED]** - YYYY-MM-DD

## Context
[Describe the forces at play and the problem being solved]

## Decision
[Describe the decision and rationale]

## Consequences
### Positive
- [List positive consequences]

### Negative
- [List negative consequences]

### Mitigation Strategies
- [List strategies to address negative consequences]

## Implementation Notes
[Technical implementation details]

## Related Decisions
[References to other ADRs]

## References
[External references and documentation]
```

## ADR Lifecycle

1. **PROPOSED**: New decision under consideration
2. **ACCEPTED**: Decision approved and implemented
3. **DEPRECATED**: Decision no longer recommended but may exist in codebase
4. **SUPERSEDED**: Decision replaced by a newer ADR

## Contributing

When making significant architectural decisions:
1. Create a new ADR using the template above
2. Discuss with the team before marking as ACCEPTED
3. Update this index with the new ADR
4. Reference the ADR in relevant code and documentation