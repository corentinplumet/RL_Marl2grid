# WCCI Tokenizer Transformer Configs

These configs run the WCCI 36 reduced-action-space MAPPO setup with the new
transformer token encoders.

The first comparison should be:

```text
wcci_group_transformer_128x3_s0.toml
wcci_entity_transformer_128x3_s0.toml
```

Then repeat for seeds 1 and 2 after the smoke runs are healthy.

Both configs keep the existing reduced action space and flat categorical action
head. Only the actor/critic encoder changes from `mlp` to `transformer`.

