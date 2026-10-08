# Binance message fixtures

These messages were **hand-constructed** from the documented Binance Spot formats (`docs/data_source_binance.md`). They were **not captured** from Binance. Prices, sizes and IDs are made up. Timestamps are 2026-10-01T12:00:00Z plus small offsets, in microseconds, except `depth_update_ms.json`, which uses milliseconds.

Each `.json` file holds one message exactly as it would arrive on the wire, as a single line of text. `malformed_cases.json` lists invalid payloads; each entry gives the expected error substring.

Unit tests read only these files and never connect to Binance.
