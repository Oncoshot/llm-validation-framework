# CHANGELOG

<!-- version list -->

## v1.0.0 (2026-08-24)

### Bug Fixes

- Score quote-stripped list cells set-wise instead of as one string (ONC-12563)
  ([`af4a411`](https://github.com/Oncoshot/llm-validation-framework/commit/af4a4119129d963f2174029d50dfdc1bc107e4fa))

### Testing

- Assert a macro F1 was actually reported, not merely present (ONC-12563)
  ([`c61c8d9`](https://github.com/Oncoshot/llm-validation-framework/commit/c61c8d9cf7d0381654faca746e232bcb163d1acd))

### Breaking Changes

- `flatten_structured_result` no longer accepts `remove_quotes`; a caller still passing it gets a
  TypeError rather than silently different scoring. Reported metrics move for anyone whose list
  cells were written in the stripped form — upward, and correct, but they move.


## v0.8.0 (2026-08-20)

### Bug Fixes

- Correct to_sortable_date return type and docstring claims
  ([`6f22573`](https://github.com/Oncoshot/llm-validation-framework/commit/6f2257321204d0146be42cebb8ae9c161cae54c7))

- Range-check the clock and stop punctuation costing the day
  ([`2cac096`](https://github.com/Oncoshot/llm-validation-framework/commit/2cac096668494e2d71bf097b49c5dcf7b6facd58))

- Stop reading an hour as a 2-digit year, and honour hour cues
  ([`e62939d`](https://github.com/Oncoshot/llm-validation-framework/commit/e62939df6e2e2f2800a7b03a2918c2963f2a487e))

### Features

- Add to_sortable_date for normalising messy date strings
  ([`7e4031c`](https://github.com/Oncoshot/llm-validation-framework/commit/7e4031c91e61dfff61dd37b1e4af319927614b1e))


## v0.7.1 (2026-07-29)

### Bug Fixes

- ONC-12379 restore semantic-release version stamping of __version__
  ([`b22c998`](https://github.com/Oncoshot/llm-validation-framework/commit/b22c998edd1c242af2761c7b5e75c7442d8d50cb))


## v0.7.0 (2026-07-20)

### Bug Fixes

- Harden the public hierarchy parameter
  ([`b1a879c`](https://github.com/Oncoshot/llm-validation-framework/commit/b1a879c28fb03248f5f27f5c52f4ecb2f4418f8e))

### Features

- Add hierarchical dictionaries (parent matching)
  ([`ac157f2`](https://github.com/Oncoshot/llm-validation-framework/commit/ac157f265010e5ceebc300bf9995564f4364033b))


## v0.6.0 (2026-07-13)

### Documentation

- ONC-12314 reference medRxiv preprint in readme
  ([`101ef22`](https://github.com/Oncoshot/llm-validation-framework/commit/101ef221cc5c8fd75c727ad1d74e667be29d6750))

### Features

- ONC-12315 update docs
  ([`22e924b`](https://github.com/Oncoshot/llm-validation-framework/commit/22e924b757974d8c925d17e8a97bc7947619e1cc))


## v0.5.0 (2026-07-08)

### Chores

- ONC-12309 normalize edited src files to LF
  ([`53e12bd`](https://github.com/Oncoshot/llm-validation-framework/commit/53e12bd17aa4189c5f229938f27bfa70f283747f))

### Documentation

- ONC-12309 document optional code field (value/code facets) in readme
  ([`49aa6ff`](https://github.com/Oncoshot/llm-validation-framework/commit/49aa6ffdd97ffc80a5e1643f6eb88dd7103af102))

### Features

- ONC-12309 optional code on StructuredField scored as -value/-code facets
  ([`41a484b`](https://github.com/Oncoshot/llm-validation-framework/commit/41a484b8b837aa54c787104eeb01fa3a4117e870))

### Testing

- ONC-12309 use Title-case confidence values to match suite convention
  ([`55905f6`](https://github.com/Oncoshot/llm-validation-framework/commit/55905f63d5525ca9342ebc7e393a5ab5bf58ca1b))


## v0.4.6 (2026-07-08)

### Chores

- ONC-12308 add .gitattributes to enforce LF line endings
  ([`7d8463e`](https://github.com/Oncoshot/llm-validation-framework/commit/7d8463ee706727783f96a71bec24960b4733ea62))

### Performance Improvements

- ONC-12308 lazy-load the scorer so structured types import without pandas
  ([`1055eb8`](https://github.com/Oncoshot/llm-validation-framework/commit/1055eb8feeec0e96ad0cc4f756d45f32b5fbdbd3))

### Refactoring

- ONC-12308 cache lazily-loaded scorer attrs and expose them via __dir__
  ([`5a348f5`](https://github.com/Oncoshot/llm-validation-framework/commit/5a348f525b17c3f25077c66970bfd83c0040977e))


## v0.4.5 (2026-06-25)

### Bug Fixes

- ONC-12248 compare binary labels by value to handle numpy bool dtype
  ([`746d657`](https://github.com/Oncoshot/llm-validation-framework/commit/746d65794286043feb9be2f9d11b21f9a91fcc29))

### Documentation

- ONC-12248 add agent entry-point doc (AGENTS.md)
  ([`cae963d`](https://github.com/Oncoshot/llm-validation-framework/commit/cae963d53f0f81dd9a424e1533d1f656e62ce5cc))


## v0.4.4 (2026-06-05)

### Bug Fixes

- Remove unusable standardize module and relax dependency floors
  ([`9aef011`](https://github.com/Oncoshot/llm-validation-framework/commit/9aef011f7774fdcea8dc3e9511402dcb91488de4))


## v0.4.3 (2026-04-26)

### Bug Fixes

- Fixed ci-cd pipeline
  ([`5d48160`](https://github.com/Oncoshot/llm-validation-framework/commit/5d48160c1384fa08d1529388197a7f2f921e0fb3))

- Fixed ci-cd pipeline
  ([`1c22970`](https://github.com/Oncoshot/llm-validation-framework/commit/1c2297065a84b1f76583a4c5717173aabdd54b26))

- Fixed ci-cd pipeline
  ([`162efc3`](https://github.com/Oncoshot/llm-validation-framework/commit/162efc399cd8a005f3b8085ea18c48c4f1bfb74e))

- ONC-11999 llmvalidate import
  ([`e8ac8dd`](https://github.com/Oncoshot/llm-validation-framework/commit/e8ac8dde930a1f1c3f38a71ffbd23f4812dadcec))

- ONC-11999 llmvalidate import with dependencies
  ([`88157ce`](https://github.com/Oncoshot/llm-validation-framework/commit/88157ce839096db1f141f6e8694bccd7315ac8d8))

### Build System

- **deps**: Bump pygments from 2.19.2 to 2.20.0
  ([`6385daf`](https://github.com/Oncoshot/llm-validation-framework/commit/6385daf61e858f54f47a808d082b4873802ce36f))

- **deps**: Bump pytest from 9.0.2 to 9.0.3
  ([`0cdde72`](https://github.com/Oncoshot/llm-validation-framework/commit/0cdde7240da3aad25969527b44acdedfdd5d221a))


## v0.4.2 (2026-03-02)

### Bug Fixes

- Fixed widgets
  ([`8b3c7d4`](https://github.com/Oncoshot/llm-validation-framework/commit/8b3c7d4d1dac25ea11508785f787626760753d91))


## v0.4.1 (2026-03-02)

### Bug Fixes

- Fixed widgets
  ([`639a8d9`](https://github.com/Oncoshot/llm-validation-framework/commit/639a8d9654857d7fdd5e325759f5de42b4a6204a))


## v0.4.0 (2026-03-02)

### Features

- Added widgets
  ([`28eab7e`](https://github.com/Oncoshot/llm-validation-framework/commit/28eab7e83a55d75fe0a7143acfdcac0a02ff0d0d))


## v0.3.0 (2026-03-02)

### Features

- Changed package name
  ([`01af816`](https://github.com/Oncoshot/llm-validation-framework/commit/01af816bc1cdd6a5d1c7e415604d2d9caa584129))


## v0.2.1 (2026-02-23)

### Bug Fixes

- Cosmetic changes in readme.md
  ([`25cccb4`](https://github.com/Oncoshot/llm-validation-framework/commit/25cccb402fada04d29335167a5f8f2d1f7bc04eb))


## v0.2.0 (2026-02-23)

### Features

- Add CI calculation
  ([`a9506f5`](https://github.com/Oncoshot/llm-validation-framework/commit/a9506f55977f18127948759dd6fb8b009fae1e2e))


## v0.1.7 (2026-02-20)

### Bug Fixes

- Test 11
  ([`34592d0`](https://github.com/Oncoshot/llm-validation-framework/commit/34592d03f2d20a0cd0c435a4d98f4c0ffcfaa3e0))


## v0.1.6 (2026-02-20)

### Bug Fixes

- Test 10
  ([`091658a`](https://github.com/Oncoshot/llm-validation-framework/commit/091658ad6b86c78b25ad0cb8b99172f9b5621843))


## v0.1.5 (2026-02-20)

### Bug Fixes

- Test 9
  ([`e73997e`](https://github.com/Oncoshot/llm-validation-framework/commit/e73997e7afcbbfc4d715a791220abbc10870b445))


## v0.1.4 (2026-02-20)

### Bug Fixes

- Test 4
  ([`81b1976`](https://github.com/Oncoshot/llm-validation-framework/commit/81b19761f4f4d1629e7a004af9462425ca912e9b))

- Test 5
  ([`c775a57`](https://github.com/Oncoshot/llm-validation-framework/commit/c775a57d76074f182430439a5e64db6bb6e32190))

- Test 6
  ([`fa91403`](https://github.com/Oncoshot/llm-validation-framework/commit/fa914032d713629f215ae2bbe21302e978626dc5))

- Test 7
  ([`afcb8db`](https://github.com/Oncoshot/llm-validation-framework/commit/afcb8db8d56cfa0e488f23906620de7b3b25b885))

- Test 8
  ([`d1afc03`](https://github.com/Oncoshot/llm-validation-framework/commit/d1afc03df392cf28632eb2f3c410c45507bc8361))


## v0.1.3 (2026-02-20)

### Bug Fixes

- Test 3
  ([`9286b7a`](https://github.com/Oncoshot/llm-validation-framework/commit/9286b7adc09626dffb13843bdf9fb831b8d59564))


## v0.1.2 (2026-02-20)


## v0.1.1 (2026-02-20)


## v0.1.0 (2026-02-20)

### Bug Fixes

- Added ci-cd
  ([`11401d2`](https://github.com/Oncoshot/llm-validation-framework/commit/11401d27c3c7302cad8e4c08d1cac38839ff2c9c))

- Added ci-cd 2
  ([`77d5625`](https://github.com/Oncoshot/llm-validation-framework/commit/77d56254984b5fc72976cbc2765c093e8ee98c52))

- Added ci-cd 3
  ([`5af28a9`](https://github.com/Oncoshot/llm-validation-framework/commit/5af28a9ff5d817b505ff67e5a0c47e0d2d16c4cc))

- Added ci-cd 4
  ([`f9e4743`](https://github.com/Oncoshot/llm-validation-framework/commit/f9e47430d22bfdf083fee77ccb2088d3d6e602b5))

- Added ci-cd 5
  ([`d62c760`](https://github.com/Oncoshot/llm-validation-framework/commit/d62c76074015afa456b13f401a64e56ed62d9ac3))

- T
  ([`dae0741`](https://github.com/Oncoshot/llm-validation-framework/commit/dae07418ae4c648c237634b24b101c6262cbfdef))

- Test
  ([`7241e25`](https://github.com/Oncoshot/llm-validation-framework/commit/7241e253567f3f8a07b5b323bc22d6e931b7aa1a))

- Test
  ([`51972f5`](https://github.com/Oncoshot/llm-validation-framework/commit/51972f5cf5bf8220b87c3fc8d9f259009f48cc1b))

- Test commit
  ([`90f97d1`](https://github.com/Oncoshot/llm-validation-framework/commit/90f97d126547269cec3f2151cba0ad793fd4d02a))

### Features

- Test
  ([`24797d7`](https://github.com/Oncoshot/llm-validation-framework/commit/24797d7b672ff385e6d512b85bb66f5088873aa2))

- Test
  ([`dda395b`](https://github.com/Oncoshot/llm-validation-framework/commit/dda395bcb78feb69347b409ae6d15a9a8908a033))

- Test
  ([`5e88fff`](https://github.com/Oncoshot/llm-validation-framework/commit/5e88fffd4b8a2cf8c752717602a1833d435ae915))


## v0.0.1 (2026-02-10)

- Initial Release
