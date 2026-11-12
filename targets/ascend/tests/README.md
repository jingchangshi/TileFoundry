# Native mainline tests

Legacy Ascend tests were archived to `../_archive_historical/legacy_tests/` by CLEAN-CODE and must not be reintroduced as compatibility assertions.

MAIN-A/B/C add normal-entry tests that run `checked_program → analyze → finalize`, require the actual installed `module.entry_function()`, and never skip/xfail the normal path.
