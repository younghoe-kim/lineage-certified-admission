# External benchmark: MaDI-Bench

The external study (pipeline step `external`, `pipeline/07_external.py`) audits the five base tasks of
MaDI-Bench, the Mannheim Data Integration Benchmark. The benchmark files are not part of this artifact;
fetch them from the official repository at the pinned commit.

| item | value |
|---|---|
| official repository | https://github.com/wbsg-uni-mannheim/MaDI-Bench |
| Git remote | https://github.com/wbsg-uni-mannheim/MaDI-Bench.git |
| branch the commit was taken from | `main` |
| pinned commit | `062058d09c8abd97026e6d4e9908b433753f58d3` (committed 2026-10-09) |
| pinned tree | `05f6a54f48eafb9d1f08d249eea82e66a77d161d` |
| audited scope | `use cases/<domain>/base/input/` for games, companies, music, products, papers |
| audited files | 133 files, 130,653,392 bytes |

Benchmark citation (from the repository's `CITATION.cff`): A. Steiner, R. Peeters, C. Bizer,
"MaDI-Bench: An End-to-End Data Integration Benchmark", 2026, arXiv:2606.30371.

The step also compares the audit with the benchmark's normalization gold, the files
`use cases/<domain>/base/input/normalization/{test,validation}.csv` of games, companies and music (part of the
audited files below). `outputs/external/gold_check.csv` holds counts per query and
`outputs/external/gold_check.json` the comparison rule and the SHA-256 of the six gold files; no benchmark value
is written.

## Fetch and verify

```sh
python3 -I data/external/fetch_external.py          # clones into data/external/MaDI-Bench and verifies
python3 -I pipeline/07_external.py                  # runs the external step on that checkout
```

`fetch_external.py` clones the official repository without file contents, checks out the pinned commit
with only the five audited input folders materialised, and then verifies

1. the origin URL, the commit id and the tree id;
2. every audited file against the blob id recorded in the pinned tree and against the SHA-256 list below;
3. the five per-domain input hashes (below), which bind the full list of paths, sizes and SHA-256 values.

To verify a copy you already have (for example a checkout of the commit, or a folder of files plus a
Git directory that holds the pinned commit), run

```sh
python3 -I data/external/fetch_external.py --checkout PATH [--git-dir GITDIR]
```

It lists every missing or differing file and exits with status 1 unless all 133 files match.
`pipeline/07_external.py` performs the same verification before it reads any benchmark file; with
`--allow-partial` (and a Git directory) it runs on an incomplete copy, leaves out the rows that need a
missing file and lists the missing files in `outputs/external/source_freeze.json`.

## Why the files are not redistributed

At the pinned commit the repository has no top-level licence file (no `LICENSE*`, `LICENCE*`, `COPYING*`
or `NOTICE*`), and neither `CITATION.cff` nor `pyproject.toml` declares a licence; the only `LICENSE`
files sit under `usecases_synthetic/third_party/` and cover bundled third-party code. No licence grants
the right to redistribute the benchmark data, so this artifact carries only metadata and hashes. The public
outputs keep the minimal source-declared metadata needed to reproduce the audit: the names of the audited source
columns, target attributes and taxonomy columns and the declared aliases of the six audited attributes, with counts
and hashes (`outputs/external/query_registry.csv`; `source_scope.csv` and `transform_authorities.csv` give the gold
correspondences of the 14 audited source columns to target attributes). Benchmark rows, free-text notes, source
metadata, taxonomies and schema-mapping files are not redistributed: outputs that copy benchmark content (for example
`outputs/external/taxonomy_rows.jsonl` and the free-text reasons in `outputs/external/unmapped_source_columns.csv`) are
not shipped; the step regenerates them from a verified checkout. This metadata is attributed to the
pinned MaDI-Bench release and is not relicensed by this artifact.

## Per-domain input hashes

SHA-256 of the canonical JSON list `[[relative_path, size_bytes, sha256], ...]` of a domain's files
(paths in code-point order, compact separators, UTF-8, trailing newline).

| domain | input hash |
|---|---|
| games | `d82201f478a3f2f43224ef949934ade390740725b8f25b78c90a8b049161f5c9` |
| companies | `efd374b29669c06d075e4e1ddcea38d629b3db3678cddf24716393b0c9636d06` |
| music | `ee9a7b9f578949435ac20d63244bd17f206acdb72f3d0a883a3e19bdc7161955` |
| products | `ddefb8e307f091ed4853e630fc60974f773414cface8f1586f7b0ae9cad61aaf` |
| papers | `4b694015fb15b5c49b0c0e36914f9ea88a21c64d6a2f16d3ede2f7772e4d9f09` |

## Per-file SHA-256 list

`sha256sum` format, paths relative to the repository root. `outputs/external/source_inventory.csv`
carries the same list with sizes and Git-LFS flags once the step has run on a complete checkout.

<!-- sha256-list:begin -->
```text
c3e154fdd068d939b1724b57b4b2cd84b157a977a532646004a206cd2b418a37  use cases/companies/base/input/data/dbpedia.csv
beb102f80e8f592f47b300590a23049f3f888ea7d384193e87b0e828a747e9cf  use cases/companies/base/input/data/dbpedia_metadata.json
1b95f9a611e6500aac69bf7dea27c1e1aa2b067181cea069158119da0f6580f3  use cases/companies/base/input/data/forbes.csv
63b806ff8acb6e14d71708345a182ed75b00cfbf69a51515c9772ec5455db497  use cases/companies/base/input/data/forbes_metadata.json
40947bea53f0ccbc8b0e159d6a0b4081678fb087c946ab8605f2d7ff621449a3  use cases/companies/base/input/data/fullcontact.csv
4834fae0c70200950abe3281b17e7041d84eddcd3443cdb7453bfd930840cdc2  use cases/companies/base/input/data/fullcontact_metadata.json
98ef70de0be6dfb34824559230ddb5fa01175239135bc073fdd018ffb22c865c  use cases/companies/base/input/entitymatching/forbes_2_dbpedia_all.csv
cf7f2476c8002324f3f19676791f500e72a86de7020211188dc8627ae25d8874  use cases/companies/base/input/entitymatching/forbes_2_dbpedia_test.csv
9d8839dc8c1be1bca17e3d6f6f06752c0f53a646caba3db50cdfbda4d2f9d4b2  use cases/companies/base/input/entitymatching/forbes_2_dbpedia_train.csv
4eff6ebcac443864ae42ba230b3e2dd2f07c53ce72e31871b60a500a4696928b  use cases/companies/base/input/entitymatching/forbes_2_dbpedia_val.csv
618aec5229b8ee66b851090e25d6f9d4732c5334a542f9619965db5fd29d138b  use cases/companies/base/input/entitymatching/forbes_2_fullcontact_all.csv
16097ceaa87df47644993f1e64226ac9af851c7c9c17b46cd46a09439d3fae12  use cases/companies/base/input/entitymatching/forbes_2_fullcontact_test.csv
85beaa268ccadb17e0f1f07917bc1d00d59b4bac6bff376853a7413efe7347c8  use cases/companies/base/input/entitymatching/forbes_2_fullcontact_train.csv
fbfa59bfab0f2fa813c5d4a06ce87ae2373dfd0127a41bd8d94feeabc341c53e  use cases/companies/base/input/entitymatching/forbes_2_fullcontact_val.csv
81db67b6a5702b9b68f0016f061c409bf3fb16d062fc854d1b424bb4e9c28c56  use cases/companies/base/input/fusion/GOLD_VERSION
00903eadb922d8b78070b8b62bddbdfe3955f385bfa4fbab0e9cf2a704a9fd07  use cases/companies/base/input/fusion/README.md
90edbefed23406e610790747ca178e2dcfa61df545ff3e882ed1b6ece96e7aa3  use cases/companies/base/input/fusion/test_set.xml
1125f3da26364570c3f0a5909770218507b3d6bd2fee926ad5e28264f59c9b44  use cases/companies/base/input/fusion/test_set_better_readability.csv
80aed44a7a5a98d1054c3a11d598b7600d80dc0e84645cd052206c10145b6149  use cases/companies/base/input/fusion/validation_set.xml
4f54244969ee087abeac24124040450d532eb31d1a535097144006b526a0f7e4  use cases/companies/base/input/fusion/validation_set_better_readability.csv
1d35401aa901faf09fe9b7707fbccaed7959c1ed739ce63d9aa82bc06aa9b24c  use cases/companies/base/input/normalization/test.csv
09e44d987d94880815020168e5e8d674e41a1bd55fc379fcf00d9643a512cf0e  use cases/companies/base/input/normalization/validation.csv
ee9ef8ea3e6f0ed4150e1d608f247ec33ca4de3697bbbb58506bcafadf22c7d1  use cases/companies/base/input/schemamatching/CLDR_Country_Taxonomy.csv
6753135c3f2854101167bc4724742d8c1eabbef43756215a59f04592d343027a  use cases/companies/base/input/schemamatching/GICS_Industry_Taxonomy.csv
e43ee4bdc6bbcbd60edee930d3e8847477d8892fd51d78d90311a8eb4df7ca01  use cases/companies/base/input/schemamatching/sm_mapping_gold.json
2b198a554bab5d2f72fcf6e5972d3855e9ac47bd1ae88ccacebbb1a6c5601e81  use cases/companies/base/input/schemamatching/target_schema.json
e7c7e4677e8dc95b245c83ab320428b5ec3fca65217143a7c90e7871221ea4c0  use cases/games/base/input/data/dbpedia.csv
8b2c41fb1ac65d0c4c30148a0b7064040c9e59ea192ab604bb2205f124d2de91  use cases/games/base/input/data/dbpedia_metadata.json
08e1dfa5d8c00a9196a0c4de285eac44977acb37f95fdbdd4c56ebd8064a572e  use cases/games/base/input/data/metacritic.csv
e118e92ca3d09293a12c92a1904d4fc2035c202bbdce151940d0ce3a95bd735f  use cases/games/base/input/data/metacritic_metadata.json
184d4c4fc93c0775b5f283c6cda7118f4fe5bf22fa4ca579ada06701dc3cc5fd  use cases/games/base/input/data/sales.csv
dd166af088f64da45d7b138c4640ce494d388e21492fe85b375d7181401a4b40  use cases/games/base/input/data/sales_metadata.json
23489ac2d83c5549b17b190424c2518c8563dc30f52e732a30763f386a521206  use cases/games/base/input/entitymatching/dbpedia_2_metacritic_test.csv
3e494118ae684aca911aec455787604f4b10a58d1bd549b2957df28b2a565c6f  use cases/games/base/input/entitymatching/dbpedia_2_sales_test.csv
90f05e242b3eb85ed57b06ce56e8b735727de87d11a29468b3032c08e2f7fb53  use cases/games/base/input/entitymatching/dbpedia_2_sales_train.csv
6c36771baace0ceb8dcb11859c355978d6aea4e0719e95b487bf265a8591c23a  use cases/games/base/input/entitymatching/dbpedia_2_sales_val.csv
c4d001f2e7a4511702a2f43d2cbb82bdae891d08d846d6eaba80498b61bb11f9  use cases/games/base/input/entitymatching/metacritic_2_dbpedia_train.csv
cfa99dd44d9944a0dd86962cdefd83ba7bc1f5b308d34d8f3e55fec79a95cbc8  use cases/games/base/input/entitymatching/metacritic_2_dbpedia_val.csv
81db67b6a5702b9b68f0016f061c409bf3fb16d062fc854d1b424bb4e9c28c56  use cases/games/base/input/fusion/GOLD_VERSION
450b0e1713bbf51a677f7584ffbb474480afc22def2cfc7baec36a99161e4e70  use cases/games/base/input/fusion/README.md
99206041ed6c97c923b165cc109e03ca94a7c5254489038a0166c3ca281296bb  use cases/games/base/input/fusion/test_set.xml
183e35facf9b675be05a7112ed5a4e08ba4f20630cf5eba52648581d721bfd2e  use cases/games/base/input/fusion/test_set_better_readability.csv
00ea7561ada8c4ba281cf4b22afd20c33da4ff90e6e5d5acfd5c2524c04fd8f2  use cases/games/base/input/fusion/validation_set.xml
b5f133d9fc07e57ea6122e7bfb54c0955f9dc2e8bf6e93b08fa27e4a88798408  use cases/games/base/input/fusion/validation_set_better_readability.csv
af923edc874345fe8f7c620394d1cc8944991ef4c709e67938bf53f031d8b097  use cases/games/base/input/normalization/test.csv
22247a34d78ac677a06a47d674280b94b3bdbbe6a5974230258ecf5463e84b09  use cases/games/base/input/normalization/validation.csv
3068183234cae431e9cf3d1ad39604c180bf751202cb270b0f7130dea7775ea1  use cases/games/base/input/schemamatching/ESRB_Rating_Taxonomy.csv
5c50b18e39696cb7389ce39de50369d208e3d67fdf07f3da2dc7a40cea49f572  use cases/games/base/input/schemamatching/Gaming_Platforms_Taxonomy.csv
9239ac6da7c1fc772b1c98f624c22bcf77550e434e5d2a18757bc57f5d5b4917  use cases/games/base/input/schemamatching/Video_Game_Genres_Taxonomy.csv
4dd6e0f099879d803b8a4257ebd482fb56874a1d23baafe5700e17b84549c01b  use cases/games/base/input/schemamatching/sm_mapping_gold.json
db3357047845d8920e7f4195d903fc8811a2fc01ae8b118633a6ebac3569ec70  use cases/games/base/input/schemamatching/target_schema.json
0fab4d11a01d8ef2db38a2b7ae98279f351583bba2459e21e0c62c778f3e3a9e  use cases/music/base/input/data/discogs.csv
8bf68d92f86f90fb9c23cfbe0174377dacfe2362063457c7cf5fe411b91a0221  use cases/music/base/input/data/discogs_metadata.json
5c12950dc9ecda3c097f98de64302f32f2f9ae0c25f72ac483ee2446a007984c  use cases/music/base/input/data/lastfm.csv
6e055ed1a5e680547022bd51f3d35527e449a26de1a90898a40bab6b9becde98  use cases/music/base/input/data/lastfm_metadata.json
746c14c4ef6d0f7a4882272501591ad91c3ac2ff4552dff57656fdb2dc53445a  use cases/music/base/input/data/musicbrainz.csv
c95d00e8e38c7d216b9725f11d8e4459366ad31e7431444d0f5eada150034b03  use cases/music/base/input/data/musicbrainz_metadata.json
a02f6f4463a387ec47c52e9f6409a6478c51cc8e8ad700958f4f856fa1617cf8  use cases/music/base/input/entitymatching/musicbrainz_2_discogs.csv
f21dd0fe2f82c91587aa3edb4117d69b148e957e0f39fa185a55f72c60ea620c  use cases/music/base/input/entitymatching/musicbrainz_2_discogs_test.csv
882848526c886959c0995abc4cc0a746d4c917735fa6053bf65690f55dee3ca3  use cases/music/base/input/entitymatching/musicbrainz_2_discogs_train.csv
40c64f645b791d60b4a0f6b86923cc50ec27533a6f257ba825c9f4fc4ae9aee0  use cases/music/base/input/entitymatching/musicbrainz_2_discogs_train_small.csv
ee7f52015ac47900f4096bd113f5dfd999debdfdc1d5bda260541371acfa137d  use cases/music/base/input/entitymatching/musicbrainz_2_discogs_val.csv
8a675e5401871e5b3eea174ce1478def8eee29c7cb53c4e30d8612b23a18c022  use cases/music/base/input/entitymatching/musicbrainz_2_lastfm.csv
7da92f87f6aeb574987ea697a1a9fc2dac2279f62a4dd7362723d7a7581d225e  use cases/music/base/input/entitymatching/musicbrainz_2_lastfm_test.csv
b2665839e294ca66be4b4bacae717bcaee766809b1e134630720a7afcf269fbc  use cases/music/base/input/entitymatching/musicbrainz_2_lastfm_train.csv
6863400e8b59f9abe2b6668aa27c32a428dadc7509d1568d1976266bbd81d191  use cases/music/base/input/entitymatching/musicbrainz_2_lastfm_train_small.csv
e9915dbeca1ce38c24851c320a6692f40a4489b72cb9a5923ce8e574474c913e  use cases/music/base/input/entitymatching/musicbrainz_2_lastfm_val.csv
81db67b6a5702b9b68f0016f061c409bf3fb16d062fc854d1b424bb4e9c28c56  use cases/music/base/input/fusion/GOLD_VERSION
450b0e1713bbf51a677f7584ffbb474480afc22def2cfc7baec36a99161e4e70  use cases/music/base/input/fusion/README.md
04fdc0d6317853e3e7312023d34177e536413e0b48a0cd296850977e9d797e6a  use cases/music/base/input/fusion/test_set.xml
274ce9e0be6a663e9a43821775b154772fad468c9c871fe8e3c300a26aad27c3  use cases/music/base/input/fusion/test_set_better_readability.csv
c063301f01734169dec2ccebae58afbf85341e3d775f49e28fd624aa483f8ea8  use cases/music/base/input/fusion/validation_set.xml
7eecd9078b50c3103695e0b2547d09f6edf30f73270dde1bae408788d4eab0d9  use cases/music/base/input/fusion/validation_set_better_readability.csv
3a17cdfd958806a0c5c3e80ca58907b143019d0e84c604808977fa00796e8b36  use cases/music/base/input/normalization/test.csv
d9c66bf70f7209a673d67e0931941a4d22fea5215df7aba35e648ac06a528f6c  use cases/music/base/input/normalization/validation.csv
be785bdc1941ab83bae13ecef63637bca3079ad573d9bc52711f758a9a624a26  use cases/music/base/input/schemamatching/Music_Genres_Taxonomy.csv
e94c90c39b3d82456eb4d9e4320143f14be23e531ee817dbfa5e68f776463361  use cases/music/base/input/schemamatching/sm_mapping_gold.json
ac167d52a0623b2c8b78282f5cf97f7e6c5b6f3f17119c3d7c18a5cf7cbb12ec  use cases/music/base/input/schemamatching/target_schema.json
5cf23872c9e62fac835312aa969291f1897186cfed764fc2fc401d4aeb333530  use cases/papers/base/input/data/crossref.jsonl
cd9a2930d3143b57a02ee64658cb71e56e226093ea12969d1077d695b3c76845  use cases/papers/base/input/data/crossref_metadata.json
644a2c3ee63787e9745127c40c16dc6d903296dadd19760db9ece86aa10ebdf6  use cases/papers/base/input/data/dblp.jsonl
6d92bb27fd5107e76d2ab01b4d8c8d13e380b7e7f6fe1059da600726b41efac9  use cases/papers/base/input/data/dblp_metadata.json
1c3941b2af896eb45c0671d779333849bd3035dc749bdd23a9a381a4c23e7ed9  use cases/papers/base/input/data/open_alex.jsonl
d6b9b90ec98aca28ed538b4b4d098481493d54a99257a5e4bedbd404c814eee1  use cases/papers/base/input/data/open_alex_metadata.json
193370bae19b96929fbc69ebed6c06f38d0bc88bc26134ef3edbb03754f1549b  use cases/papers/base/input/entitymatching/dblp_crossref_test.csv
67ce047951c17806bdadfd3eab04f66412f1f977958cbbeac6d84ea2eee118f8  use cases/papers/base/input/entitymatching/dblp_crossref_train.csv
b1f7e57bb18d2378069c7cc5e2d4d859111dfa3afdf3772e301af1ae8a3a7576  use cases/papers/base/input/entitymatching/dblp_crossref_val.csv
c505718e9e0601c75ba11f0524461b0ad0f8c142ede5e87b88c89c8ba8a812ec  use cases/papers/base/input/entitymatching/dblp_openalex_test.csv
6a1a4b50926c0218ea2c2412e857bbf31f235ecc570a346412dc5443ae095648  use cases/papers/base/input/entitymatching/dblp_openalex_train.csv
36fa9f19a20babd061517e86869a35f1240842ac729aa9d855e2d5a8828f68b1  use cases/papers/base/input/entitymatching/dblp_openalex_val.csv
81db67b6a5702b9b68f0016f061c409bf3fb16d062fc854d1b424bb4e9c28c56  use cases/papers/base/input/fusion/GOLD_VERSION
af60f329f339a3e99d26a0b74e4cb9d4a75e619b6f8b042ac71b0e8662cc0e7d  use cases/papers/base/input/fusion/README.md
b7cd0e52740531ee951e123da8905a3bb82c2d712cbadd82816c665d25cf8765  use cases/papers/base/input/fusion/fusion_test.jsonl
560d3eb45620ea285ccb88e4a88f5362a048cef702a1d34432c948e9e071d058  use cases/papers/base/input/fusion/fusion_test_better_readability.csv
2ec0028dfde4f50c97e2b64e60d39d5a7dd1ba3d9cc47c07d38de47991223cdf  use cases/papers/base/input/fusion/fusion_val.jsonl
5e6e23c300154bcb91d5befad07604f3369ad1d8d8c33263f0d509a53f0e9ca0  use cases/papers/base/input/fusion/fusion_val_better_readability.csv
d7c0a1788223b730cc777ead9c1ff49723c06ea245e25c1a4e0e836e0d0b6eb8  use cases/papers/base/input/normalization/test.csv
51229bfbb8ded53d56502c8d4fd0e12408a4b8bb1c76c6706d2e3caf3523f14d  use cases/papers/base/input/normalization/validation.csv
585fc57a9f10fe64597f1a6b0c51a3d4e19aecbbfd1168b924a90db5ebc618a7  use cases/papers/base/input/schemamatching/sm_mapping_gold.json
d1a6c65493ea3850e5d44d75d10b0919bc48b61e33ac6365a0a0c068f37be0ab  use cases/papers/base/input/schemamatching/target_schema.json
f8e7fec38b4b89ae2f203f6349db3894384caae7f7a99bb03993a96beb452b19  use cases/products/base/input/data/dataset_1.json
5487fa49475fe66496419d975942d9e95e171a928fe2fd77a27929f1f40b95af  use cases/products/base/input/data/dataset_1_metadata.json
be7457fb3350267b77c7eb88f877fae716c79f7af676f992cba5f8f34f5ad3bb  use cases/products/base/input/data/dataset_2.json
840c07418dcdcbae808459d4ea86f528f3199f999426ddd0270125df18d06180  use cases/products/base/input/data/dataset_2_metadata.json
f8e98de7045b492ed869c1c051561331e5df2c70fc76200a4dd57fb903902a75  use cases/products/base/input/data/dataset_3.json
73742346ddb14ea1861ca0c4e510facfde8adee66db0305d749e935d365530c6  use cases/products/base/input/data/dataset_3_metadata.json
be007d13284c4727f21b6855ce3a6160a7a17f746ee27e430c85454a78a9ddc2  use cases/products/base/input/data/dataset_4.json
f2e9658e652b32d3e5620b769bf24eefe2bb5311a5bc26d9956a195ad615de7f  use cases/products/base/input/data/dataset_4_metadata.json
61a45be3673af70fe1e41f924e4e2c5bf2891543712f40da0e62870c4f5ca6d1  use cases/products/base/input/entitymatching/prod1_to_prod2_all.csv
55fdc00243da221f826f9f9c0de179eec6a40a64c0e8f64bef3a029cfb89e086  use cases/products/base/input/entitymatching/prod1_to_prod2_test.csv
0cdf603ba81b810330816b80ee126d970aadee05388b26c16e40249cac4fe318  use cases/products/base/input/entitymatching/prod1_to_prod2_train.csv
5cab2bccffd167cec83d1aadb18e1750b350f9c3beae4ce6b16b1e3005a2b6df  use cases/products/base/input/entitymatching/prod1_to_prod2_val.csv
68b086fde1a95edc164fa6c28dbda63e97fe4a69ee8d86f315007c8a05fb732e  use cases/products/base/input/entitymatching/prod1_to_prod3_all.csv
f31b0411e5ac5fb90c475e52e68ac23b734b7a7c3b6a70049037df4692d867d1  use cases/products/base/input/entitymatching/prod1_to_prod3_test.csv
00e776ab28a9d3901bab411c2c23be26e8970ce65bf01b6b97513e2196fb18ef  use cases/products/base/input/entitymatching/prod1_to_prod3_train.csv
178a6c3a6f01e904802bef5dcf4535d06704846a01c573623b1c32f64d30210d  use cases/products/base/input/entitymatching/prod1_to_prod3_val.csv
924586a0754319510e564ffbee8f84fc07e65b349612d4432040cec3040f99d8  use cases/products/base/input/entitymatching/prod1_to_prod4_all.csv
2d55738c43c32a9fa50ecda5c2a0f712dc214f41bb665dbb1f82914897b612f4  use cases/products/base/input/entitymatching/prod1_to_prod4_test.csv
5723d63c186c2d5e0cd0ec638f5415ee01eac9998b49abe368a992e4fdd07d0c  use cases/products/base/input/entitymatching/prod1_to_prod4_train.csv
1241f38a63659612c62bb993be596717512fb3b23a97e12decdc81552eaf2d3a  use cases/products/base/input/entitymatching/prod1_to_prod4_val.csv
81db67b6a5702b9b68f0016f061c409bf3fb16d062fc854d1b424bb4e9c28c56  use cases/products/base/input/fusion/GOLD_VERSION
bfd2c646090b8a9b32aef4a0334da54ab777e418e632a0b3d1c6e43c796b791c  use cases/products/base/input/fusion/README.md
fdd1c3eee9d3c7c2e6ed331a7ab9ac1e46c7028814f68d0f6526867ffdcc04e0  use cases/products/base/input/fusion/fusion_test_set.csv
1c4e9297f3f23f3aded85f13c2c7a0889858ddf6bfc6c0a85d72c692bb66c892  use cases/products/base/input/fusion/fusion_test_set_better_readability.csv
8c5708f54db941a09e92d69e3455df53d01dbc12f6323b4338ca312515a2ed4f  use cases/products/base/input/fusion/fusion_validation_set.csv
0c626088f5b9690fb0cb89116f0da0c29a94be0d6207f9986e731637a92f933d  use cases/products/base/input/fusion/fusion_validation_set_better_readability.csv
81073f8726bb242fb9e42dbf19ed87ef8a4f1bf01fada4e1d8dd6e92a4c5ba7f  use cases/products/base/input/normalization/test.csv
dda855154e4d8641f95249170e0714d1d503085bb7a601e5073c4b1fd07adcdf  use cases/products/base/input/normalization/validation.csv
258bf6a1a86aa0f0ef732734d3853d3184ebccd8b2ebef51047479a3bad84b8f  use cases/products/base/input/schemamatching/GPU_Memory_Taxonomy.csv
d7d7b2b6c55408850a543b4b497e88ea8e5345fd90cca03223356a08b1af2d14  use cases/products/base/input/schemamatching/Product_Type_Taxonomy.csv
9ec7830f1382a608cca3abe93600dd0d85421e551a4a59c045110e0b12dd73ef  use cases/products/base/input/schemamatching/Storage_Interface_Taxonomy.csv
490e06133d7ffafab9167feaf95cf7d523434e8c54dc1f89ebd37607e0d0af5a  use cases/products/base/input/schemamatching/sm_mapping_gold.json
88e30ec050fd7871ffc6214c74b1b2cd43d4d8c2c72e7ab88b3b344683cb6d5b  use cases/products/base/input/schemamatching/target_schema.json
```
<!-- sha256-list:end -->
