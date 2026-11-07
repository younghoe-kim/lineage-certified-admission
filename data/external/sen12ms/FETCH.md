# SEN12MS study: Path B and witness pairs for the released 17-class scene label

The SEN12MS step (`pipeline/07d_sen12ms.py`, paper Sec. VI-C) reads four public files that are not part of this
artifact. `fetch_sen12ms.py` downloads them into `data/external/sen12ms/files/` and checks each against the SHA-256 and
the git blob id below (the same values as `sources.json`, which the step reads).

| file | repository | commit | path in the repository | bytes | SHA-256 | git blob |
|---|---|---|---|---|---|---|
| `IGBP_probability_labels.pkl` | https://github.com/schmitt-muc/SEN12MS | `3a41236a28d08d253ebe2fa1a081e5e32aa7eab4` | `labels/IGBP_probability_labels.pkl` | 41,487,095 | `3a04fabf7a83e4aee331d77727ff427804c576fd40477b40a76c31f3411aed85` | `811a117215f20f12b3e1b2a21185db0b0870079d` |
| `single_label_IGBPfull_ClsNum.txt` | https://github.com/schmitt-muc/SEN12MS | `3a41236a28d08d253ebe2fa1a081e5e32aa7eab4` | `labels/single_label_IGBPfull_ClsNum.txt` | 6,364,395 | `c0fb8842db5214ac804a5e3e55f515872afded470a23aa7a8283393f9ca3b5e7` | `9bf6b9f8e95d02ef59dedc51f5c437ea956e0baf` |
| `single_label_IGBPsimple_ClsNum.txt` | https://github.com/schmitt-muc/SEN12MS | `3a41236a28d08d253ebe2fa1a081e5e32aa7eab4` | `labels/single_label_IGBPsimple_ClsNum.txt` | 6,270,251 | `953c3e9c4c01cf748d6fb677c158ca2ffd1ff8c92449742d7cb17789628c5a6f` | `26884146f05899c6ed5ee5645c2b8d62a147bf1c` |
| `dataset.py` | https://github.com/schmitt-muc/SEN12MS | `3a41236a28d08d253ebe2fa1a081e5e32aa7eab4` | `classification/dataset.py` | 14,536 | `797b91597fddf4f09782daff92770eca66328db4b7e73de4110cf57a075cb403` | `8a4348c10bca882d704a92b9f644a06d38014d8c` |

- **Counts.** The class-fraction file holds, for each of the 180,662 patches, the fractions of the 17 IGBP land-cover
  classes over its 256 x 256 pixels. Multiplied by 65,536 they are integers that sum to 65,536 in every patch; the step
  checks this and uses the integer counts as the basis block.
- **Declared map.** `igbp_simplified_map.json` records the published merge of the 17 classes into 10 (Schmitt and
  Wu, 2021, Table 2). The step reads the official loader `dataset.py` as text, never runs it, and stops unless its
  merge (lines 68-72) is the same grouping. The loader's single label is the arg-max of the counts.
- **Query.** The view is the 10 merged counts, which the loader computes from the same fractions for the simplified
  10-class label. The protected query is the released 17-class label, one `patch key: class` line per patch in
  `single_label_IGBPfull_ClsNum.txt`; the step checks that it is the strict arg-max of the 17 counts on every patch,
  and that the released 10-class label (`single_label_IGBPsimple_ClsNum.txt`) is the strict arg-max of the view.
- **Patch keys.** Both label files key a patch by its Sentinel-2 file name, for example
  `ROIs1158_spring_s2_100_p101.tif`; the step writes witness pairs with these keys.
- **Reading the pickle.** The step and the independent check read the class-fraction file with a restricted unpickler
  that admits only NumPy array reconstruction.

## License

SEN12MS is released under CC BY 4.0. This artifact fetches its files at the pinned commit rather than copying them;
`outputs/external/` holds aggregate counts and the released keys of four witness pairs.
