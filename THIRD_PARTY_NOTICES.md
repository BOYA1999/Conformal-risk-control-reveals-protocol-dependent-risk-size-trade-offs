# Third-party notices

The MIT License in this repository applies only to repository-authored code and documentation.

Datasets, pretrained models, and external software are not relicensed here. In particular:

- B-XAIC is reported by its dataset card as CC BY-SA 4.0.
- The Graph Attribution repository is Apache-2.0; users must also follow its NOTICE and any underlying data terms.
- Polaris reports the `adme-fang-1` dataset as CC BY 4.0 and cites the source publication listed in `DATA_SOURCES.md`.
- The CheMeleon checkpoint is not redistributed because the cited Zenodo record did not display a license value when this package was prepared.
- Python dependencies keep their upstream licenses.
- MoleculeACE code is MIT at the frozen commit identified in `DATA_SOURCES.md`.
- MoleculeACE benchmark records derive from ChEMBL v29. ChEMBL and A7 data-derived outputs remain under CC BY-SA 3.0 and are not covered by this repository's MIT License.
- The Pharmaco-Explainer code repository is MIT at commit `f40599677e52574b86d8e7ccc6dc842471519cbd`; its source tree is not bundled. The upstream MIT notice is retained beside the repository-authored pharmacophore adapter scripts.
- The K4_2ar tables and released checkpoint come from `klimczakjakubdev/pharmaco-explainer` at commit `fb38e88acd4b486a4ba9df74dccfa701e05d01b1`, whose dataset card declares CC BY 4.0. Raw tables, structures, labels, molecule identifiers, predictions, atom outputs, and the checkpoint are not bundled. Released aggregate summaries retain attribution and are outside the root MIT scope.

No third-party repository snapshot, raw molecular table, or model weight is included in this release.
