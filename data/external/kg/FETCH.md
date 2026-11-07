# Link-prediction study: WN18 and WN18RR against the WordNet table of reflexive pointers

The link-prediction step (`pipeline/07c_kg.py`, paper Sec. VI-C, extended audit details Section C) reads public files that are
not part of this artifact. `fetch_kg.py` downloads them into `data/external/kg/files/` and checks every file
against the SHA-256 below (the same values as `sources.json`, which the step reads).

| file | release | repository | commit | path in the repository | bytes | SHA-256 |
|---|---|---|---|---|---|---|
| `WN18/train.txt` | WN18 | https://github.com/ibalazevic/TuckER | `23aff592115ee91e099ce490cdbea28ef026fc54` | `data/WN18/train.txt` | 4,758,553 | `24c9b02dc57d3ae8ca8467ce355995edee7be8eddc3bd070ee34d3c6fc3c95eb` |
| `WN18/valid.txt` | WN18 | https://github.com/ibalazevic/TuckER | `23aff592115ee91e099ce490cdbea28ef026fc54` | `data/WN18/valid.txt` | 169,033 | `13347bfb5e17d1cae003e0bfe29b4716d3ab748e8cfb6af997138d37da3a51dc` |
| `WN18/test.txt` | WN18 | https://github.com/ibalazevic/TuckER | `23aff592115ee91e099ce490cdbea28ef026fc54` | `data/WN18/test.txt` | 168,947 | `d22c19d176d4ba6a6f0d0c6bfce39ab3c16e3b20e1df6f9efe0d9ed86542fa10` |
| `conve/WN18RR.tar.gz` | WN18RR | https://github.com/TimDettmers/ConvE | `f3c0eb286025410fa5b4c04696c918264163d0ca` | `WN18RR.tar.gz` | 892,861 | `1eb9152f804c140d163462c0d49f65e5a0b30ab5fbb375c5e3353c11a6249060` |
| `conve/create_WN18RR.py` | WN18RR | https://github.com/TimDettmers/ConvE | `f3c0eb286025410fa5b4c04696c918264163d0ca` | `create_WN18RR.py` | 1,450 | `450fcadf1f5b82325487912d4323bcee9272d0aa4adc993de3c3b92467eaaae7` |
| `wordnet/wordnet.zip` | WordNet 3.0 | https://github.com/nltk/nltk_data | `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a` | `packages/corpora/wordnet.zip` | 10,775,600 | `cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59` |
| `WN18/wordnet-mlj12-definitions.txt` | WN18 | https://github.com/ibalazevic/TuckER | `23aff592115ee91e099ce490cdbea28ef026fc54` | `data/WN18/wordnet-mlj12-definitions.txt` | 4,414,562 | `70d85bc995d4f82c6c1038d6a3869a149e50622fbdf846af3ab45c519521f627` |

- **WN18** (Bordes et al., 2013) is read from the TuckER repository, whose copy of the splits is byte-identical to
  the copy in the DeepGraphLearning/KnowledgeGraphEmbedding repository (`data/wn18/`, commit
  `2e440e0f9c687314d5ff67ead68ce985dc446e3a`).
- **WN18RR** (Dettmers et al., 2018) is read from the archive of the ConvE repository. The step reads the three split
  files in memory and extracts nothing.
- **Construction script.** `create_WN18RR.py` of the ConvE repository is read as text. The step parses its list
  literal `predicates_to_remove` and never runs the script. Applied to the WN18 splits, that relation filter
  reproduces all three WN18RR splits triple for triple (`outputs/kg/wn18rr_filter_check.json`).
- **WordNet 3.0** is read from the NLTK data repository (`wordnet.zip`, with `wordnet/LICENSE`). The step reads
  the four database files `wordnet/data.{noun,verb,adj,adv}`. A synset is an offset together with a part of
  speech; WN18 identifiers are offsets only, and WN18 merges synsets that share an offset across parts of speech.
  The check therefore gives each entity of a certified triple and its training witness one part of speech under
  which both are pointers between the same two synsets (`outputs/kg/raw_authority.json`).
- **WN18 definitions** (`wordnet-mlj12-definitions.txt`, same TuckER commit) name one word with its part of speech
  for each WN18 entity. The step uses that part of speech for one diagnostic only: how many certified pairs also
  match when every entity takes it, and whether each other pair involves an offset shared by two parts of speech.

## Registry

`wordnet_reflexive.yaml` transcribes the table of reflexive pointers of the WordNet documentation, wninput(5WN),
https://wordnet.princeton.edu/node/30 (read on 2026-10-29), row by row. A row "Pointer → Reflect" declares that a
pointer of type Pointer from x to y implies a pointer of type Reflect from y to x. The Domain row is stated in one
direction only, and the registry keeps it that way. Also-see pointers have no row and stay outside the registry.
No row is mined from the benchmark files, and no model score is read.

## Licenses

WN18 and WN18RR derive from WordNet 3.0 and are subject to its license (Princeton University; the notice is in
`wordnet/LICENSE` of the archive above). The repositories that host them, TuckER and ConvE (including the construction
script), are under the MIT license. The WordNet 3.0 database is read from the NLTK data repository, whose wordnet
archive carries the WordNet license. The construction script is read as text, never run. This artifact does not
redistribute these files; `outputs/kg/` holds counts only.
