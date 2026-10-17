# Presets

```python
import bipgen

bipgen.presets.available()                 # ['c58_16', 'c58_4', 'c58_5']
spec = bipgen.presets.load("c58_4")
graph = bipgen.generate(spec)              # about half a minute on one core
table = bipgen.compare(bipgen.presets.observed("c58_4"), [graph])
```

--8<-- "src/bipgen/presets/README.md"
