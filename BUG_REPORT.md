---
name: Bug report
about: Create a report to help us improve YFlow
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## Bug Description

**Clear description of the bug**
A concise explanation of what the bug is.

## Environment

- **OS**: [e.g., Windows 11, macOS 14, Ubuntu 22.04]
- **Python Version**: [e.g., 3.9.7, 3.10.12, 3.11.5]
- **YFlow Version**: [e.g., 0.3.0, commit hash if using dev version]
- **Device**: [CPU / GPU (specify model if GPU)]
- **Dependencies**: [numpy version, cupy version if using GPU]

## Steps to Reproduce

Please provide minimal code to reproduce the issue:

```python
# Example:
from yflow.yformers.model import DecoderOnlyModel
import numpy as np

# Setup
model = DecoderOnlyModel(vocab_size=1000, d_model=128, num_heads=4, d_ff=512, num_layers=2)
input_tokens = np.random.randint(0, 1000, (2, 10))

# This causes the bug
output = model(input_tokens)
```

**Steps:**
1. Import module '...'
2. Initialize '...'
3. Call method '...'
4. Observe error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened instead.

## Error Message / Stack Trace

```
Paste the complete error message and stack trace here
```

## Additional Context

**Screenshots**
If applicable, add screenshots to help explain the problem.

**Workarounds**
If you found a temporary workaround, please share it.

**Related Issues**
Link to any related issues or PRs.

**Impact**
- [ ] Blocks production use
- [ ] Blocks development
- [ ] Minor inconvenience
- [ ] Documentation issue

## Possible Solution (Optional)

If you have ideas about what might be causing the bug or how to fix it, please share.

---

**Before submitting:**
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproducible example
- [ ] I have included my environment details
- [ ] I have included the complete error message
